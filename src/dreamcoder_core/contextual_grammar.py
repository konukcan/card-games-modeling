"""
ContextualGrammarNetwork: Context-aware primitive prediction for enumeration guidance.

This module implements DreamCoder-style contextual grammar prediction, where
the probability of a primitive depends on:
1. The task embedding (what task we're solving)
2. The parent primitive (what function we're filling an argument for)
3. The argument position (which argument slot)

This provides much more informative guidance than global primitive prediction,
because the useful primitives depend heavily on context. For example:
- Under 'filter', we often want predicates like 'eq', 'gt', 'lt'
- Under 'map', we often want transformations like 'get_rank', 'get_suit'
- At root level, we often want 'filter', 'map', 'fold'

References:
    - DreamCoder paper (PLDI 2021): Section on recognition models
    - ellisk42/ec repository: recognition.py, ContextualGrammarNetwork classes

TODO(enumeration-integration): Integrate ContextualGrammarNetwork with TopDownEnumerator
    This requires modifying TopDownEnumerator to:
    1. Accept a ContextualGrammarNetwork instead of just a Grammar
    2. Call network.predict(task_embedding, parent_idx, arg_position) at each choice point
    3. Use predicted log-probs to bias primitive selection during enumeration

    Integration steps:
    1. Modify enumeration.py:TopDownEnumerator.enumerate() to track parent context
    2. Add contextual_grammar parameter to TopDownEnumerator.__init__()
    3. Replace grammar.productions[prim].log_probability with contextual predictions
    4. Update run_overnight_v3.py to pass trained ContextualGrammarNetwork to enumerator
    5. Add training for ContextualGrammarNetwork in wake-sleep loop

    See: experiments/evaluate_head_improvements.py for initial testing framework
"""

from typing import Dict, List, Tuple, Optional, Set
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamcoder_core.program import Program, Primitive, Invented, Application, Abstraction, Index
from dreamcoder_core.grammar import Grammar


class ContextualGrammarNetwork(nn.Module):
    """
    Context-aware primitive prediction network.

    Predicts P(primitive | task, parent, arg_position) for contextual
    enumeration guidance.

    Architecture:
        Context = Concat[τ, Embed(parent), Embed(position)]
        Output = LogSoftmax(MLP(Context))

    Two variants available:
    - 'full': Separate prediction for each context (more expressive, more params)
    - 'mask': Shared task-conditional prediction + learned context biases (efficient)
    """

    # Special context indices
    NO_PARENT_IDX = -1  # Root level
    VARIABLE_PARENT_IDX = -2  # Lambda variable as parent

    def __init__(
        self,
        grammar: Grammar,
        task_dim: int = 32,
        parent_embed_dim: int = 16,
        position_embed_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        max_positions: int = 4,
        dropout: float = 0.1,
        variant: str = 'mask',  # 'full' or 'mask'
        device: str = 'cpu'
    ):
        """
        Initialize ContextualGrammarNetwork.

        Args:
            grammar: Grammar object with primitives
            task_dim: Dimension of task embedding (from ContrastiveRecognitionModel)
            parent_embed_dim: Dimension of parent primitive embedding
            position_embed_dim: Dimension of argument position embedding
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            max_positions: Maximum argument position (typically 3-4)
            dropout: Dropout rate
            variant: 'full' or 'mask' (see class docstring)
            device: Device for tensors
        """
        super().__init__()

        self.device = device
        self.variant = variant
        self.max_positions = max_positions

        # Build primitive vocabulary
        self.primitive_names = [str(p.program) for p in grammar.productions]
        self.primitive_to_idx = {name: i for i, name in enumerate(self.primitive_names)}
        self.num_primitives = len(self.primitive_names)

        # Build library structure: maps primitive to list of argument context indices
        self.library = self._build_library(grammar)
        self.n_contexts = sum(len(args) for args in self.library.values()) + 2  # +2 for root, variable

        # Context index mapping
        self.context_to_idx = self._build_context_index_map()

        # Embeddings for parent primitive (including special indices)
        # We use num_primitives + 2 to handle NO_PARENT and VARIABLE_PARENT
        self.parent_embedding = nn.Embedding(self.num_primitives + 2, parent_embed_dim)
        self.position_embedding = nn.Embedding(max_positions + 1, position_embed_dim)

        # Combined input dimension
        context_dim = task_dim + parent_embed_dim + position_embed_dim

        if variant == 'full':
            # Full model: separate MLP for each context (more expressive)
            self.mlp = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                *[layer for _ in range(num_layers - 2) for layer in [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ]],
                nn.Linear(hidden_dim, self.num_primitives)
            )
        else:
            # Mask variant: shared task prediction + learned context biases
            # This is more parameter-efficient
            self.task_head = nn.Sequential(
                nn.Linear(task_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_primitives)
            )
            # Unconditional context biases (transition matrix)
            self.context_biases = nn.Parameter(
                torch.zeros(self.n_contexts, self.num_primitives)
            )
            nn.init.xavier_uniform_(self.context_biases)

        self.to(device)

    def _build_library(self, grammar: Grammar) -> Dict[str, List[int]]:
        """
        Build library mapping primitives to their argument context indices.

        Each primitive with arity n gets n context indices, one for each
        argument position.

        Returns:
            Dict mapping primitive name to list of context indices
        """
        library = {}
        current_idx = 2  # Start after root (0) and variable (1) contexts

        for prod in grammar.productions:
            prim_name = str(prod.program)
            # Get arity from type
            tp = prod.tp
            arity = 0
            while hasattr(tp, 'arguments') and tp.arguments:
                arity += 1
                tp = tp.returns if hasattr(tp, 'returns') else tp

            if arity > 0:
                library[prim_name] = list(range(current_idx, current_idx + arity))
                current_idx += arity
            else:
                library[prim_name] = []

        return library

    def _build_context_index_map(self) -> Dict[Tuple, int]:
        """
        Build mapping from (parent_name, position) to context index.
        """
        context_map = {
            (None, 0): 0,  # Root context
            ('$', 0): 1,   # Variable parent context
        }

        for prim_name, indices in self.library.items():
            for pos, idx in enumerate(indices):
                context_map[(prim_name, pos)] = idx

        return context_map

    def get_context_index(
        self,
        parent_name: Optional[str],
        position: int
    ) -> int:
        """
        Get context index for a (parent, position) pair.

        Args:
            parent_name: Name of parent primitive, None for root, '$' for variable
            position: Argument position (0-indexed)

        Returns:
            Context index for this context
        """
        if parent_name is None:
            return 0  # Root
        if parent_name.startswith('$'):
            return 1  # Variable

        key = (parent_name, min(position, self.max_positions - 1))
        return self.context_to_idx.get(key, 0)  # Default to root if unknown

    def forward(
        self,
        task_embedding: torch.Tensor,
        parent_idx: torch.Tensor,
        position_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict primitive log-probabilities given context.

        Args:
            task_embedding: (batch, task_dim) task embeddings
            parent_idx: (batch,) index of parent primitive
                       (0 to num_primitives-1, or num_primitives for root,
                        num_primitives+1 for variable)
            position_idx: (batch,) argument position (0 to max_positions)

        Returns:
            (batch, num_primitives) log-probabilities
        """
        # Clamp indices to valid ranges
        parent_idx = parent_idx.clamp(0, self.num_primitives + 1)
        position_idx = position_idx.clamp(0, self.max_positions)

        parent_emb = self.parent_embedding(parent_idx)
        position_emb = self.position_embedding(position_idx)

        if self.variant == 'full':
            context = torch.cat([task_embedding, parent_emb, position_emb], dim=-1)
            logits = self.mlp(context)
        else:
            # Mask variant: base prediction + context bias
            base_logits = self.task_head(task_embedding)

            # Get context index for bias lookup
            batch_size = task_embedding.shape[0]
            context_indices = self._compute_context_indices(parent_idx, position_idx)
            context_bias = self.context_biases[context_indices]

            logits = base_logits + context_bias

        return F.log_softmax(logits, dim=-1)

    def _compute_context_indices(
        self,
        parent_idx: torch.Tensor,
        position_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute context indices from parent and position indices.

        This maps (parent, position) pairs to flat context indices.
        """
        batch_size = parent_idx.shape[0]
        context_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i in range(batch_size):
            p_idx = parent_idx[i].item()
            pos_idx = position_idx[i].item()

            if p_idx == self.num_primitives:
                # Root context
                context_indices[i] = 0
            elif p_idx == self.num_primitives + 1:
                # Variable context
                context_indices[i] = 1
            else:
                # Normal primitive parent
                prim_name = self.primitive_names[p_idx] if p_idx < len(self.primitive_names) else None
                if prim_name and prim_name in self.library:
                    arg_indices = self.library[prim_name]
                    if pos_idx < len(arg_indices):
                        context_indices[i] = arg_indices[pos_idx]
                    else:
                        context_indices[i] = 0  # Fallback to root
                else:
                    context_indices[i] = 0

        return context_indices

    def predict_for_context(
        self,
        task_embedding: torch.Tensor,
        parent_name: Optional[str],
        position: int
    ) -> torch.Tensor:
        """
        Convenience method for single-context prediction.

        Args:
            task_embedding: (task_dim,) single task embedding
            parent_name: Name of parent primitive (None for root)
            position: Argument position

        Returns:
            (num_primitives,) log-probabilities
        """
        # Convert to batch format
        tau = task_embedding.unsqueeze(0)

        # Get parent index
        if parent_name is None:
            parent_idx = torch.tensor([self.num_primitives], device=self.device)
        elif parent_name.startswith('$'):
            parent_idx = torch.tensor([self.num_primitives + 1], device=self.device)
        else:
            idx = self.primitive_to_idx.get(parent_name, self.num_primitives)
            parent_idx = torch.tensor([idx], device=self.device)

        position_idx = torch.tensor([min(position, self.max_positions)], device=self.device)

        log_probs = self.forward(tau, parent_idx, position_idx)
        return log_probs.squeeze(0)

    def expand_for_invention(self):
        """
        Expand the network to handle a new invented primitive.

        Called when library learning discovers a new abstraction.
        """
        old_num = self.num_primitives
        new_num = old_num + 1

        # Expand parent embedding
        old_embed = self.parent_embedding
        new_embed = nn.Embedding(new_num + 2, old_embed.embedding_dim)
        new_embed.weight.data[:old_num + 2] = old_embed.weight.data
        nn.init.xavier_uniform_(new_embed.weight.data[old_num + 2:])
        self.parent_embedding = new_embed

        # Expand output layer
        if self.variant == 'full':
            old_out = self.mlp[-1]
            new_out = nn.Linear(old_out.in_features, new_num)
            new_out.weight.data[:old_num] = old_out.weight.data
            new_out.bias.data[:old_num] = old_out.bias.data
            nn.init.xavier_uniform_(new_out.weight.data[old_num:])
            new_out.bias.data[old_num:] = 0.0
            self.mlp[-1] = new_out
        else:
            old_out = self.task_head[-1]
            new_out = nn.Linear(old_out.in_features, new_num)
            new_out.weight.data[:old_num] = old_out.weight.data
            new_out.bias.data[:old_num] = old_out.bias.data
            nn.init.xavier_uniform_(new_out.weight.data[old_num:])
            new_out.bias.data[old_num:] = 0.0
            self.task_head[-1] = new_out

            # Expand context biases
            new_biases = nn.Parameter(torch.zeros(self.n_contexts, new_num))
            new_biases.data[:, :old_num] = self.context_biases.data
            self.context_biases = new_biases

        self.num_primitives = new_num

    def to(self, device):
        """Move network to device."""
        self.device = device
        return super().to(device)


def extract_contextual_training_data(
    program: Program,
    primitive_to_idx: Dict[str, int]
) -> List[Tuple[Optional[str], int, str]]:
    """
    Extract (parent_name, position, child_name) tuples from a program.

    Used to create training data for the ContextualGrammarNetwork.

    Example:
        (filter (λ (eq (get_suit $0) HEARTS)))

    Extracts:
        (None, 0, 'filter')           # filter at root
        ('filter', 0, 'λ')            # lambda is filter's 1st arg
        ('eq', 0, 'get_suit')         # get_suit is eq's 1st arg
        ('eq', 1, 'HEARTS')           # HEARTS is eq's 2nd arg
        ('get_suit', 0, '$0')         # $0 is get_suit's arg

    Args:
        program: Program to extract contexts from
        primitive_to_idx: Mapping from primitive names to indices

    Returns:
        List of (parent_name, position, child_name) tuples
    """
    contexts = []

    def visit(node: Program, parent_name: Optional[str], position: int):
        if isinstance(node, (Primitive, Invented)):
            name = str(node)
            contexts.append((parent_name, position, name))
            # Primitives have no children

        elif isinstance(node, Application):
            # Unroll application chain
            head = node
            args = []
            while isinstance(head, Application):
                args.insert(0, head.x)
                head = head.f

            if isinstance(head, (Primitive, Invented)):
                head_name = str(head)
                contexts.append((parent_name, position, head_name))

                # Visit arguments with head as parent
                for arg_pos, arg in enumerate(args):
                    visit(arg, head_name, arg_pos)
            else:
                # Head is lambda or variable
                visit(node.f, parent_name, position)
                visit(node.x, parent_name, position + 1)

        elif isinstance(node, Abstraction):
            # Lambda - pass through parent context
            visit(node.body, parent_name, position)

        elif isinstance(node, Index):
            # Variable - record context
            var_name = f'${node.i}'
            contexts.append((parent_name, position, var_name))

    visit(program, None, 0)
    return contexts


def build_contextual_training_batch(
    programs: List[Program],
    task_embeddings: List[torch.Tensor],
    primitive_to_idx: Dict[str, int],
    num_primitives: int,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build training batch for ContextualGrammarNetwork.

    Args:
        programs: List of solved programs
        task_embeddings: Corresponding task embeddings
        primitive_to_idx: Mapping from primitive names to indices
        num_primitives: Total number of primitives
        device: Device for tensors

    Returns:
        Tuple of:
        - task_embs: (N, task_dim) repeated task embeddings
        - parent_indices: (N,) parent primitive indices
        - position_indices: (N,) argument positions
        - target_indices: (N,) target primitive indices
    """
    all_task_embs = []
    all_parent_indices = []
    all_position_indices = []
    all_target_indices = []

    for prog, tau in zip(programs, task_embeddings):
        contexts = extract_contextual_training_data(prog, primitive_to_idx)

        for parent_name, position, child_name in contexts:
            # Skip variables as targets (they're not in our vocabulary)
            if child_name.startswith('$'):
                continue

            if child_name not in primitive_to_idx:
                continue

            all_task_embs.append(tau)

            # Parent index
            if parent_name is None:
                all_parent_indices.append(num_primitives)  # Root
            elif parent_name.startswith('$'):
                all_parent_indices.append(num_primitives + 1)  # Variable
            else:
                all_parent_indices.append(primitive_to_idx.get(parent_name, num_primitives))

            all_position_indices.append(min(position, 3))
            all_target_indices.append(primitive_to_idx[child_name])

    if not all_task_embs:
        # Return empty tensors
        task_dim = task_embeddings[0].shape[-1] if task_embeddings else 32
        return (
            torch.zeros(0, task_dim, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device)
        )

    return (
        torch.stack(all_task_embs).to(device),
        torch.tensor(all_parent_indices, dtype=torch.long, device=device),
        torch.tensor(all_position_indices, dtype=torch.long, device=device),
        torch.tensor(all_target_indices, dtype=torch.long, device=device)
    )

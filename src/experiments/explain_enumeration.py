#!/usr/bin/env python3
"""
Concrete walkthrough of partial programs and enumeration.
Shows WHY grammar weights affect programs/second.
"""

print("""
================================================================================
PARTIAL PROGRAMS AND ENUMERATION EXPLAINED
================================================================================

WHAT IS A PARTIAL PROGRAM?
--------------------------
A partial program has "holes" that need to be filled. Each hole has a type.

Example: We want a program of type (Hand → Bool)

Step 0: Start with a single hole
  Program: ?0:(Hand→Bool)
  Cost: 0.0

  The "?" means "this needs to be filled"

Step 1: Fill the hole with a lambda (to handle the Hand argument)
  Program: λ. ?1:Bool
  Cost: 0.0 (lambdas are free - they're structural)

  Now we have one argument $0:Hand bound, and need a Bool

Step 2: Fill ?1 with a primitive, e.g., "all_same_color"

  If all_same_color has log_prob = -2.0 in our grammar:
    Program: λ. (all_same_color ?2:Hand)
    Cost: 0.0 + 2.0 = 2.0

  If all_same_color has log_prob = -0.5 (high probability):
    Cost: 0.0 + 0.5 = 0.5

Step 3: Fill ?2 with $0 (the bound variable)
  If variable has log_prob = -1.0:
    Final Program: λ. (all_same_color $0)
    Total Cost: 2.0 + 1.0 = 3.0

================================================================================
THE PRIORITY QUEUE
================================================================================

The enumerator maintains a MIN-HEAP priority queue:
  - Items are sorted by COST (lower = better = explored first)
  - Cost = -log_probability (so high prob → low cost)

At each step:
  1. POP the lowest-cost partial program
  2. EXPAND its first hole with all possible productions
  3. PUSH new partial programs back with updated costs

Example priority queue state:
  [
    (cost=0.5, λ. (all_same_color ?:Hand)),      # Explored FIRST
    (cost=2.0, λ. (has_suit ?:Hand ?:Suit)),     # Explored later
    (cost=3.5, λ. (le ?:Int ?:Int)),             # Explored much later
    ...
  ]

================================================================================
HOW GRAMMAR WEIGHTS AFFECT SPEED
================================================================================

Consider two scenarios:

SCENARIO A: Uniform grammar (all primitives equal probability)
  - 59 primitives, each with prob ≈ 1/59 ≈ 0.017
  - log_prob ≈ -4.08 for each primitive

  When expanding a hole, we create 59 branches, ALL with similar costs.
  The priority queue grows WIDE → lots of partial programs to track.

  Priority queue might look like:
    [(4.08, prog1), (4.08, prog2), (4.08, prog3), ..., (4.08, prog59)]

  We explore all of them before going deeper. Queue size explodes!

SCENARIO B: Focused grammar (recognition model assigns high prob to useful prims)
  - 5 primitives with prob ≈ 0.15 each → log_prob ≈ -1.9
  - 54 primitives with prob ≈ 0.005 each → log_prob ≈ -5.3

  When expanding a hole:
    - 5 branches have low cost (1.9), explored FIRST
    - 54 branches have high cost (5.3), PRUNED or explored much later

  Priority queue might look like:
    [(1.9, prog1), (1.9, prog2), ..., (1.9, prog5), (5.3, prog6), ...]

  We explore the 5 promising paths deeply before touching the 54 unpromising ones.
  Queue stays NARROW → faster heap operations, less memory.

================================================================================
WHY DOES THIS AFFECT PROGRAMS/SECOND?
================================================================================

1. HEAP OPERATIONS:
   - heappush/heappop are O(log n) where n = queue size
   - Focused grammar → smaller queue → faster operations
   - 10x smaller queue → roughly log(10) = 3x fewer comparisons per operation

2. PRUNING:
   The enumerator has cost bounds (e.g., max_depth * 10).
   Low-probability primitives accumulate high cost quickly and get PRUNED:

     if new_cost > budget * 2:  # Skip expensive paths
         continue

   With focused grammar, more paths hit the prune threshold early.

3. CACHE EFFICIENCY:
   Smaller queue = better cache utilization = faster memory access

4. FEWER PARTIAL PROGRAMS:
   The key metric is NOT complete programs enumerated, but PARTIAL programs explored.
   Focused grammar explores fewer partial programs to find the same solutions.

================================================================================
CONCRETE NUMBERS
================================================================================

From your experiments:
  Neural iteration 1: 111,664 programs in 42 min = 44 prog/s
  Neural iteration 6: 343,000 programs in 40 min = 143 prog/s  (3x faster!)

Why the speedup?
  - After training on more solved tasks, neural model focuses probability mass
  - Priority queue stays smaller, pruning happens earlier
  - Same timeout, more complete programs yielded

Contrastive models stayed flat because their predictions never became more focused
(due to the τ ≈ 0 problem we found).
""")

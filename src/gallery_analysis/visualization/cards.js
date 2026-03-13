/**
 * cards.js — Card rendering module for the gallery visualization.
 *
 * Provides functions to render individual cards, hands, and rule-level
 * hand grids from JSON data produced by cards.py.
 *
 * Usage:
 *   const { renderCard, renderHand, renderRuleHands, renderTestHand } = CardRenderer;
 */
const CardRenderer = (function () {
  /**
   * Create an <img> element for a single card.
   *
   * @param {Object} card - Card data with image_path, suit, rank.
   * @returns {HTMLImageElement} The card image element.
   */
  function renderCard(card) {
    const img = document.createElement("img");
    img.src = card.image_path;
    img.alt = card.rank + " of " + card.suit;

    // Standard card proportions: height 110px, width scaled by 2.5/3.5
    // (playing cards have roughly a 5:7 aspect ratio)
    img.style.height = "110px";
    img.style.width = Math.round(110 * (2.5 / 3.5)) + "px";

    return img;
  }

  /**
   * Render a row of cards (one hand) into a container element.
   *
   * @param {Array<Object>} handData - Array of card objects.
   * @param {HTMLElement} containerEl - DOM element to render into.
   */
  function renderHand(handData, containerEl) {
    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.gap = "6px";
    row.style.padding = "6px";
    row.style.backgroundColor = "#f5f5f5";
    row.style.borderRadius = "4px";

    handData.forEach(function (card) {
      row.appendChild(renderCard(card));
    });

    containerEl.appendChild(row);
  }

  /**
   * Render all 6 primary hands for a rule, stacked vertically
   * with "Hand 1" ... "Hand 6" labels.
   *
   * @param {Object} handsData - Object with a "hands" array (from hands_to_json).
   * @param {HTMLElement} containerEl - DOM element to render into.
   */
  function renderRuleHands(handsData, containerEl) {
    var hands = handsData.hands || handsData;

    hands.forEach(function (hand, index) {
      var wrapper = document.createElement("div");
      wrapper.style.marginBottom = "8px";

      var label = document.createElement("div");
      label.textContent = "Hand " + (index + 1);
      label.style.fontWeight = "bold";
      label.style.fontSize = "13px";
      label.style.marginBottom = "2px";
      wrapper.appendChild(label);

      renderHand(hand, wrapper);
      containerEl.appendChild(wrapper);
    });
  }

  /**
   * Render a test hand with P(accept), confidence, and correctness badge.
   *
   * @param {Array<Object>} handData - Array of card objects with image_path.
   * @param {Object} metrics - {p_accept, confidence, ground_truth, correct_prediction}.
   * @param {HTMLElement} containerEl - DOM element to render into.
   */
  function renderTestHand(handData, metrics, containerEl) {
    var wrapper = document.createElement("div");
    wrapper.style.marginBottom = "10px";

    // Render the card images
    renderHand(handData, wrapper);

    // Metrics line below the cards
    var info = document.createElement("div");
    info.style.fontSize = "0.78rem";
    info.style.color = "#555";
    info.style.marginTop = "3px";
    info.style.paddingLeft = "6px";

    var pAccept = metrics.p_accept !== undefined ? metrics.p_accept.toFixed(3) : "—";
    var conf = metrics.confidence !== undefined ? metrics.confidence.toFixed(3) : "—";
    var gtLabel = metrics.ground_truth ? "Accept" : "Reject";
    var correctIcon = metrics.correct_prediction ? "✓" : "✗";
    var correctColor = metrics.correct_prediction ? "#2CA02C" : "#C44E52";

    info.innerHTML =
      "P(accept): <strong>" + pAccept + "</strong> &nbsp;|&nbsp; " +
      "Conf: <strong>" + conf + "</strong> &nbsp;|&nbsp; " +
      "GT: " + gtLabel + " " +
      "<span style='color:" + correctColor + ";font-weight:bold;'>" + correctIcon + "</span>";

    wrapper.appendChild(info);
    containerEl.appendChild(wrapper);
  }

  /**
   * Render a category of test hands with a "Sample" button that shows one
   * random hand at a time, plus an "Expand all" toggle to reveal every hand.
   *
   * Layout per category:
   *   [Category Label (n)]  [Sample] [Expand all v]
   *   <single sampled hand with metrics>
   *   <collapsed list of all hands, toggled by Expand>
   *
   * @param {string} label - Category label (e.g. "Easy ACCEPT").
   * @param {string} color - CSS color for the label.
   * @param {Array<Object>} hands - Array of hand objects with .hand (cards) and metrics.
   * @param {HTMLElement} containerEl - DOM element to render into.
   */
  function renderTestHandCategory(label, color, hands, containerEl) {
    if (!hands || hands.length === 0) return;

    // ── Header row: label + buttons ──
    var header = document.createElement("div");
    header.style.display = "flex";
    header.style.alignItems = "center";
    header.style.gap = "8px";
    header.style.marginTop = "12px";
    header.style.marginBottom = "6px";

    var heading = document.createElement("span");
    heading.textContent = label + " (" + hands.length + ")";
    heading.style.fontWeight = "bold";
    heading.style.fontSize = "0.82rem";
    heading.style.color = color;
    header.appendChild(heading);

    // Shared button style helper
    function styleBtn(btn) {
      btn.style.fontSize = "0.72rem";
      btn.style.padding = "2px 8px";
      btn.style.border = "1px solid #ccc";
      btn.style.borderRadius = "4px";
      btn.style.background = "#fff";
      btn.style.cursor = "pointer";
      btn.style.color = "#555";
    }

    var sampleBtn = document.createElement("button");
    sampleBtn.textContent = "Sample";
    styleBtn(sampleBtn);
    header.appendChild(sampleBtn);

    var expandBtn = document.createElement("button");
    expandBtn.textContent = "Show all \u25BC";
    styleBtn(expandBtn);
    header.appendChild(expandBtn);

    containerEl.appendChild(header);

    // ── Sample display area (shows one hand) ──
    var sampleArea = document.createElement("div");
    containerEl.appendChild(sampleArea);

    // Show initial random sample
    function showSample() {
      sampleArea.innerHTML = "";
      var idx = Math.floor(Math.random() * hands.length);
      var entry = hands[idx];
      renderTestHand(entry.hand, entry, sampleArea);
    }
    showSample();
    sampleBtn.addEventListener("click", showSample);

    // ── Expandable area (all hands, hidden by default) ──
    var expandArea = document.createElement("div");
    expandArea.style.display = "none";
    expandArea.style.borderLeft = "3px solid " + color;
    expandArea.style.paddingLeft = "8px";
    expandArea.style.marginTop = "6px";
    containerEl.appendChild(expandArea);

    var expanded = false;
    expandBtn.addEventListener("click", function () {
      expanded = !expanded;
      if (expanded) {
        expandBtn.textContent = "Hide all \u25B2";
        // Render all hands if not already rendered
        if (expandArea.children.length === 0) {
          hands.forEach(function (entry, i) {
            var lbl = document.createElement("div");
            lbl.textContent = "#" + (i + 1);
            lbl.style.fontSize = "0.72rem";
            lbl.style.color = "#999";
            lbl.style.marginTop = "6px";
            expandArea.appendChild(lbl);
            renderTestHand(entry.hand, entry, expandArea);
          });
        }
        expandArea.style.display = "block";
      } else {
        expandBtn.textContent = "Show all \u25BC";
        expandArea.style.display = "none";
      }
    });
  }

  // Public API
  return {
    renderCard: renderCard,
    renderHand: renderHand,
    renderRuleHands: renderRuleHands,
    renderTestHand: renderTestHand,
    renderTestHandCategory: renderTestHandCategory,
  };
})();

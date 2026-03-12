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
   * PLACEHOLDER: Render a test hand with associated metrics.
   * Not yet implemented — logs a warning.
   *
   * @param {Array<Object>} handData - Array of card objects.
   * @param {Object} metrics - Associated metrics for the hand.
   * @param {HTMLElement} containerEl - DOM element to render into.
   */
  function renderTestHand(handData, metrics, containerEl) {
    console.warn("CardRenderer.renderTestHand is not yet implemented");
  }

  // Public API
  return {
    renderCard: renderCard,
    renderHand: renderHand,
    renderRuleHands: renderRuleHands,
    renderTestHand: renderTestHand,
  };
})();

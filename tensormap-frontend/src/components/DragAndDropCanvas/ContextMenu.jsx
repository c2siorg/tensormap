import PropTypes from "prop-types";

/**
 * A minimal right-click context menu for canvas nodes.
 *
 * Renders at fixed screen coordinates (clientX / clientY) so it stays
 * pinned to where the user right-clicked regardless of canvas pan/zoom.
 * A full-screen transparent overlay sits behind the menu so that any
 * click outside the menu triggers onClose without needing document-level
 * event listeners (which would require explicit cleanup).
 */
function ContextMenu({ x, y, onDuplicate, onClose }) {
  return (
    <>
      {/* transparent overlay — catches clicks outside the menu */}
      <div className="fixed inset-0 z-40" onClick={onClose} />

      <div
        style={{ left: x, top: y }}
        className="fixed z-50 min-w-[140px] rounded-md border bg-white py-1 shadow-md"
      >
        <button
          className="w-full px-3 py-1.5 text-left text-sm hover:bg-accent"
          onClick={onDuplicate}
        >
          Duplicate
        </button>
      </div>
    </>
  );
}

ContextMenu.propTypes = {
  x: PropTypes.number.isRequired,
  y: PropTypes.number.isRequired,
  onDuplicate: PropTypes.func.isRequired,
  onClose: PropTypes.func.isRequired,
};

export default ContextMenu;

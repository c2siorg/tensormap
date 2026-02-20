import { useState } from "react";
import "./KeyboardShortcutsHelp.css";

const shortcuts = [
    { keys: ["Ctrl", "Z"], description: "Undo last action" },
    { keys: ["Ctrl", "Shift", "Z"], description: "Redo last action" },
    { keys: ["Ctrl", "D"], description: "Duplicate selected node(s)" },
    { keys: ["Delete"], description: "Delete selected nodes/edges" },
    { keys: ["Backspace"], description: "Delete selected nodes/edges" },
];

function KeyboardShortcutsHelp() {
    const [open, setOpen] = useState(false);

    return (
        <div className="kbd-help-container">
            <button
                className="kbd-help-trigger"
                onClick={() => setOpen((prev) => !prev)}
                title="Keyboard Shortcuts"
                aria-label="Show keyboard shortcuts"
            >
                ?
            </button>
            {open && (
                <div className="kbd-help-panel">
                    <div className="kbd-help-header">
                        <span>Keyboard Shortcuts</span>
                        <button
                            className="kbd-help-close"
                            onClick={() => setOpen(false)}
                            aria-label="Close shortcuts panel"
                        >
                            ✕
                        </button>
                    </div>
                    <ul className="kbd-help-list">
                        {shortcuts.map(({ keys, description }) => (
                            <li key={description} className="kbd-help-item">
                                <span className="kbd-help-desc">{description}</span>
                                <span className="kbd-help-keys">
                                    {keys.map((k, i) => (
                                        <span key={k}>
                                            <kbd className="kbd-key">{k}</kbd>
                                            {i < keys.length - 1 && (
                                                <span className="kbd-plus">+</span>
                                            )}
                                        </span>
                                    ))}
                                </span>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}

export default KeyboardShortcutsHelp;

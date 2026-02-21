import { useLocation, useNavigate } from "react-router-dom";
import * as constants from "../../../constants/Strings";
import * as urls from "../../../constants/Urls";

function SideBar({ children }) {
  const location = useLocation();
  const navigate = useNavigate();

  const handleItemClick = (name) => {
    if (name === constants.HOME_SIDEBAR) {
      navigate(urls.HOME_URL);
    } else if (name === constants.DATA_UPLOAD_SIDEBAR) {
      navigate(urls.DATA_UPLOAD_URL);
    } else if (name === constants.DATA_PROCESS_SIDEBAR) {
      navigate(urls.DATA_PROCESS_URL);
    } else {
      navigate(urls.DEEP_LEARN_URL);
    }
  };

  const menuItems = [
    { name: constants.HOME_SIDEBAR, path: urls.HOME_URL },
    { name: constants.DATA_UPLOAD_SIDEBAR, path: urls.DATA_UPLOAD_URL },
    { name: constants.DATA_PROCESS_SIDEBAR, path: urls.DATA_PROCESS_URL },
    { name: constants.DEEP_LEARNING_SIDEBAR, path: urls.DEEP_LEARN_URL },
  ];

  return (
    <div>
      <div className="mt-2 flex gap-4">
        <div className="w-48 shrink-0">
          <nav className="flex flex-col gap-1">
            {menuItems.map((item) => (
              <button
                key={item.name}
                className={`rounded-md px-3 py-2 text-left text-sm font-bold transition-colors hover:bg-accent ${
                  location.pathname === item.path ? "bg-accent text-accent-foreground" : ""
                }`}
                onClick={() => handleItemClick(item.name)}
              >
                {item.name}
              </button>
            ))}
          </nav>
        </div>

        <div className="min-h-[90vh] flex-1 rounded-md border p-4">{children}</div>
      </div>
    </div>
  );
}

export default SideBar;

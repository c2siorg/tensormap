import { useLocation, useNavigate } from "react-router-dom";
import * as constants from "../../../constants/Strings";
import * as urls from "../../../constants/Urls";

function NavBar() {
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

  const navItems = [
    { name: constants.DATA_UPLOAD_SIDEBAR, path: urls.DATA_UPLOAD_URL },
    { name: constants.DATA_PROCESS_SIDEBAR, path: urls.DATA_PROCESS_URL },
    { name: constants.DEEP_LEARNING_SIDEBAR, path: urls.DEEP_LEARN_URL },
  ];

  return (
    <div>
      <nav className="flex items-center justify-between rounded-md border px-4 py-2">
        <button
          className="flex items-center gap-2 text-lg font-bold"
          onClick={() => handleItemClick(constants.HOME_SIDEBAR)}
        >
          <img src="/favicon.png" alt="logo" className="h-6 w-6" />
          TensorMap
        </button>
        <div className="flex gap-1">
          {navItems.map((item) => (
            <button
              key={item.name}
              className={`rounded-md px-3 py-2 text-sm font-bold transition-colors hover:bg-accent ${
                location.pathname === item.path ? "bg-accent text-accent-foreground" : ""
              }`}
              onClick={() => handleItemClick(item.name)}
            >
              {item.name}
            </button>
          ))}
        </div>
      </nav>
    </div>
  );
}

export default NavBar;

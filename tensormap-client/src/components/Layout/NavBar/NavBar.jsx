import { Menu } from "semantic-ui-react";
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
  return (
    <div>
      <Menu stackable size="large">
        <Menu.Item
          name={constants.HOME_SIDEBAR}
          active={location.pathname === urls.HOME_URL}
          onClick={() => handleItemClick(constants.HOME_SIDEBAR)}
          style={{ fontWeight: "bold" }}
        >
          <img src="/favicon.png" alt="logo" />
          TensorMap
        </Menu.Item>
        <Menu.Menu position="right">
          <Menu.Item
            name={constants.DATA_UPLOAD_SIDEBAR}
            active={location.pathname === urls.DATA_UPLOAD_URL}
            onClick={() => handleItemClick(constants.DATA_UPLOAD_SIDEBAR)}
            style={{ fontWeight: "bold" }}
          />
          <Menu.Item
            name={constants.DATA_PROCESS_SIDEBAR}
            active={location.pathname === urls.DATA_PROCESS_URL}
            onClick={() => handleItemClick(constants.DATA_PROCESS_SIDEBAR)}
            style={{ fontWeight: "bold" }}
          />
          <Menu.Item
            name={constants.DEEP_LEARNING_SIDEBAR}
            active={location.pathname === urls.DEEP_LEARN_URL}
            onClick={() => handleItemClick(constants.DEEP_LEARNING_SIDEBAR)}
            style={{ fontWeight: "bold" }}
          />
        </Menu.Menu>
      </Menu>
    </div>
  );
}

export default NavBar;

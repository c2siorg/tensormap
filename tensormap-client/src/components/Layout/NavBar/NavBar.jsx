import React from 'react';
import { Menu } from 'semantic-ui-react';

function NavBar() {
  return (
    <div>
      <Menu stackable size="huge">
        <Menu.Item>
          <img src="/favicon.png" alt="logo" />
        </Menu.Item>

        <Menu.Item
          name="TensorMap"
          active
          style={{ fontWeight: 'bold' }}
        />
      </Menu>
    </div>
  );
}

export default NavBar;
import React from 'react';
import NavBar from './NavBar/NavBar';
import SideBar from './SideBar/SideBar';
import { Segment } from "semantic-ui-react";

function Layout({ children }) {
  return (
    <div>
      <NavBar />
      <Segment>
        {children}
      </Segment>
    </div>
  );
}

export default Layout;

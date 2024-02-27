import React from 'react';
import NavBar from './NavBar/NavBar';
import SideBar from './SideBar/SideBar';

function Layout(props) {
  return (
    <div>
      <NavBar />
      <SideBar mainProps={props} />
    </div>
  );
}

export default Layout;

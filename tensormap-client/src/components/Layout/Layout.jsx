import React from 'react';
import NavBar from './NavBar/NavBar';
import SideBar from './SideBar/SideBar';

function Layout({ children }) {
  return (
    <div>
      <NavBar />
      <SideBar> 
        {children}
      </SideBar>
    </div>
  );
}

export default Layout;

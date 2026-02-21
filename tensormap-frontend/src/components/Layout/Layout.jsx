import NavBar from "./NavBar/NavBar";

function Layout({ children }) {
  return (
    <div>
      <NavBar />
      <div className="rounded-md border p-4">{children}</div>
    </div>
  );
}

export default Layout;

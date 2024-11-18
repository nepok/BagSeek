import React from 'react';
import './Header.css';
import { FaCog, FaSearch } from 'react-icons/fa'; // Importing icons

const Header: React.FC = () => {
  return (
    <div className="header-container">
      <div className="header-left">
        <h1 className="header-title">BagSeek</h1>
      </div>
      <div className="header-right">
        <FaSearch className="header-icon" />
        <FaCog className="header-icon" />
      </div>
    </div>
  );
};

export default Header;

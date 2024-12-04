import React from 'react';
import './Header.css'; // Make sure this file contains the @import rule
import { IconButton, Typography, Box } from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';

const Header: React.FC = () => {
  return (
    <Box className="header-container" display="flex" justifyContent="space-between" alignItems="center" >
      {/* Left section with title */}
      <Typography variant="h4" className="header-title">
        BagSeek
      </Typography>

      {/* Right section with icons */}
      <Box className="header-right" display="flex" alignItems="center">
        <IconButton className="header-icon">
          <SettingsIcon />
        </IconButton>
      </Box>
    </Box>
  );
};

export default Header;
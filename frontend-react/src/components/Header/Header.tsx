import React from 'react';
import './Header.css'; // Make sure this file contains the @import rule
import { IconButton, Typography, Box, Tooltip } from '@mui/material';
import FolderIcon from '@mui/icons-material/Folder';
import IosShareIcon from '@mui/icons-material/IosShare';
import SettingsIcon from '@mui/icons-material/Settings';

interface HeaderProps {
  setIsFloatingBoxVisible: (visible: boolean | ((prev: boolean) => boolean)) => void;
}

const Header: React.FC<HeaderProps> = ({ setIsFloatingBoxVisible }) => {
  return (
    <Box
      className="header-container"
      display="flex"
      justifyContent="space-between"
      alignItems="center"
      padding="8px 16px"
    >
      {/* Left section with title */}
      <Typography variant="h4" className="header-title">
        BagSeek
      </Typography>

      {/* Right section with icons */}
      <Box display="flex" alignItems="center">
        <Tooltip title="Open Rosbag" arrow>
          <IconButton
            className="header-icon"
            onClick={() => setIsFloatingBoxVisible((prev: boolean) => !prev)}
          >
            <FolderIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Export Rosbag" arrow>
          <IconButton className="header-icon">
            <IosShareIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Settings" arrow>
          <IconButton className="header-icon">
            <SettingsIcon />
          </IconButton>
        </Tooltip>
      </Box>
    </Box>
  );
};

export default Header;
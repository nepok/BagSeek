import React, { useState } from 'react';
import { Box, IconButton, Popover, Typography } from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

interface HelpPopoverProps {
  title: string;
  content: React.ReactNode;
  /** Extra sx applied to the trigger IconButton */
  iconSx?: object;
}

/**
 * A small ? icon button that opens a styled popover with a title and arbitrary content.
 * Designed for contextual help throughout the app.
 */
const HelpPopover: React.FC<HelpPopoverProps> = ({ title, content, iconSx }) => {
  const [anchor, setAnchor] = useState<HTMLButtonElement | null>(null);

  return (
    <>
      <IconButton
        size="small"
        onClick={(e) => { e.stopPropagation(); setAnchor(e.currentTarget); }}
        sx={{
          color: 'rgba(255,255,255,0.35)',
          p: 0.25,
          flexShrink: 0,
          '&:hover': { color: 'rgba(255,255,255,0.75)' },
          ...iconSx,
        }}
      >
        <HelpOutlineIcon sx={{ fontSize: 15 }} />
      </IconButton>

      <Popover
        open={Boolean(anchor)}
        anchorEl={anchor}
        onClose={() => setAnchor(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
        transformOrigin={{ vertical: 'top', horizontal: 'left' }}
        PaperProps={{
          sx: {
            background: '#1a1a1a',
            border: '1px solid rgba(255,255,255,0.12)',
            borderRadius: 2,
            maxWidth: 320,
            p: 2,
          },
        }}
      >
        <Typography
          variant="subtitle2"
          sx={{ color: 'white', fontWeight: 600, mb: 1, fontSize: '0.825rem' }}
        >
          {title}
        </Typography>
        <Box sx={{ color: 'rgba(255,255,255,0.72)', fontSize: '0.78rem', lineHeight: 1.6 }}>
          {content}
        </Box>
      </Popover>
    </>
  );
};

export default HelpPopover;

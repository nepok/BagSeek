import { createTheme } from '@mui/material/styles';

// Create a custom dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#B49FCC',
    },
    secondary: {
      main: '#B3D6C6',
    },
  },
  
});

export default darkTheme;
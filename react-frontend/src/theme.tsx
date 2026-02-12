import { createTheme } from '@mui/material/styles';

// Create a custom dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#B49FCC',
    },
    secondary: {
      main: '#B3D6B9',
    },
    warning: {
      main: '#E8C872',
    },
    info: {
      main: '#8BA9C8',
    },
  },
  
});

export default darkTheme;
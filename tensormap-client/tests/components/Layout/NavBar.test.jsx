import React from 'react';
import { render } from '@testing-library/react';
import NavBar from '../../../src/components/Layout/NavBar/NavBar';
import '@testing-library/jest-dom'
import { BrowserRouter } from 'react-router-dom';

describe('<NavBar />', () => {
  it('renders without crashing', () => {
    render(<BrowserRouter><NavBar/></BrowserRouter>);
  });

  it('renders the logo image correctly', () => {
    const { getByAltText } = render(<BrowserRouter><NavBar/></BrowserRouter>);
    const logoImg = getByAltText('logo');
    expect(logoImg).toBeInTheDocument();
    expect(logoImg.src).toBe('http://localhost/favicon.png');
  });
});

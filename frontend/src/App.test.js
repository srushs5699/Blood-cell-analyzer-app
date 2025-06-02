import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './App';

describe('App Component', () => {
  test('renders main application title', () => {
    render(<App />);
    expect(screen.getByText('🩸 Blood Cell Analyzer')).toBeInTheDocument();
  });

  test('renders navigation tabs', () => {
    render(<App />);
    
    expect(screen.getByRole('button', { name: /Upload & Analyze/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Dashboard/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Results/i })).toBeInTheDocument();
  });

  test('can switch to dashboard tab', () => {
    render(<App />);
    
    const dashboardTab = screen.getByRole('button', { name: /Dashboard/i });
    fireEvent.click(dashboardTab);
    
    expect(screen.getByText('📊 Analytics Dashboard')).toBeInTheDocument();
  });

  test('shows upload interface by default', () => {
    render(<App />);
    
    expect(screen.getByText('🩸 Upload Blood Smear Images')).toBeInTheDocument();
    
    // Debug: Let's see all buttons
    const buttons = screen.getAllByRole('button');
    console.log('All buttons found:', buttons.map(btn => btn.textContent));
    
    // Try different ways to find the analyze button
    const analyzeButton = buttons.find(btn => btn.textContent.includes('Analyze'));
    expect(analyzeButton).toBeInTheDocument();
  });

  test('results tab is initially disabled', () => {
    render(<App />);
    
    const resultsTab = screen.getByRole('button', { name: /Results/i });
    expect(resultsTab).toBeDisabled();
  });

  test('dashboard shows statistics', () => {
    render(<App />);
    
    // Click dashboard tab
    const dashboardTab = screen.getByRole('button', { name: /Dashboard/i });
    fireEvent.click(dashboardTab);
    
    // Check for statistics
    expect(screen.getByText('Total Analyses')).toBeInTheDocument();
    expect(screen.getByText('156')).toBeInTheDocument();
    expect(screen.getByText('Avg Processing Time')).toBeInTheDocument();
    expect(screen.getByText('1.8s')).toBeInTheDocument();
    expect(screen.getByText('Average Accuracy')).toBeInTheDocument();
    expect(screen.getByText('93.2%')).toBeInTheDocument();
  });

  test('analyze button functionality', () => {
    render(<App />);
    
    // Make sure we're on upload tab
    const uploadTab = screen.getByRole('button', { name: /Upload & Analyze/i });
    fireEvent.click(uploadTab);
    
    // Find analyze button using different methods
    const analyzeButton = screen.getByText(/Analyze Blood Cells/i);
    expect(analyzeButton).toBeInTheDocument();
    
    // Click analyze button
    fireEvent.click(analyzeButton);
    
    // Should show loading state
    expect(screen.getByText(/Analyzing/i)).toBeInTheDocument();
  });
});

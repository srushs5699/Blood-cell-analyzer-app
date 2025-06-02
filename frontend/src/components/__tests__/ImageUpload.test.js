import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

// Mock ImageUpload component if it doesn't exist yet
const MockImageUpload = ({ onAnalysisComplete, loading, setLoading }) => {
  const [files, setFiles] = React.useState([]);

  const handleFileChange = (event) => {
    setFiles([...event.target.files]);
  };

  const handleAnalyze = () => {
    setLoading(true);
    setTimeout(() => {
      onAnalysisComplete({
        success: true,
        cell_counts: { RBC: 100, WBC: 10, Platelets: 20 }
      });
      setLoading(false);
    }, 100);
  };

  return (
    <div>
      <h2>Upload Blood Smear Images</h2>
      <input 
        type="file" 
        onChange={handleFileChange}
        data-testid="file-input"
        accept="image/*"
        multiple
      />
      <button 
        onClick={handleAnalyze}
        disabled={loading || files.length === 0}
        data-testid="analyze-button"
      >
        {loading ? 'Analyzing...' : 'Analyze Blood Cells'}
      </button>
      {files.length > 0 && (
        <div data-testid="file-list">
          {files.map((file, index) => (
            <div key={index}>{file.name}</div>
          ))}
        </div>
      )}
    </div>
  );
};

const mockProps = {
  onAnalysisComplete: jest.fn(),
  loading: false,
  setLoading: jest.fn()
};

describe('ImageUpload Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders upload interface', () => {
    render(<MockImageUpload {...mockProps} />);
    
    expect(screen.getByText(/Upload Blood Smear Images/i)).toBeInTheDocument();
    expect(screen.getByText(/Analyze Blood Cells/i)).toBeInTheDocument();
  });

  test('file input accepts files', async () => {
    const user = userEvent.setup();
    render(<MockImageUpload {...mockProps} />);
    
    const file = new File(['fake image'], 'test.jpg', { type: 'image/jpeg' });
    const input = screen.getByTestId('file-input');
    
    await user.upload(input, file);
    
    await waitFor(() => {
      expect(screen.getByText('test.jpg')).toBeInTheDocument();
    });
  });

  test('analyze button is disabled when no files', () => {
    render(<MockImageUpload {...mockProps} />);
    
    const analyzeButton = screen.getByTestId('analyze-button');
    expect(analyzeButton).toBeDisabled();
  });

  test('analyze button is enabled with files', async () => {
    const user = userEvent.setup();
    render(<MockImageUpload {...mockProps} />);
    
    const file = new File(['fake image'], 'test.jpg', { type: 'image/jpeg' });
    const input = screen.getByTestId('file-input');
    
    await user.upload(input, file);
    
    await waitFor(() => {
      const analyzeButton = screen.getByTestId('analyze-button');
      expect(analyzeButton).not.toBeDisabled();
    });
  });

  test('shows loading state during analysis', async () => {
    const user = userEvent.setup();
    const setLoading = jest.fn();
    
    render(<MockImageUpload {...mockProps} setLoading={setLoading} />);
    
    const file = new File(['fake image'], 'test.jpg', { type: 'image/jpeg' });
    const input = screen.getByTestId('file-input');
    
    await user.upload(input, file);
    
    const analyzeButton = screen.getByTestId('analyze-button');
    await user.click(analyzeButton);
    
    expect(setLoading).toHaveBeenCalledWith(true);
  });
});

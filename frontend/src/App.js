import React, { useState } from 'react';
import './components/ImageUpload.css';


// Simple inline styles
const styles = {
  app: {
    textAlign: 'center',
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    fontFamily: 'Arial, sans-serif',
    padding: '20px'
  },
  header: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    padding: '2rem',
    marginBottom: '2rem',
    borderRadius: '10px'
  },
  navTabs: {
    display: 'flex',
    justifyContent: 'center',
    gap: '1rem',
    marginBottom: '2rem'
  },
  button: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    border: 'none',
    padding: '1rem 2rem',
    borderRadius: '50px',
    color: 'white',
    cursor: 'pointer'
  },
  activeButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    color: '#667eea'
  },
  content: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: '20px',
    padding: '2rem',
    color: '#333',
    maxWidth: '800px',
    margin: '0 auto'
  }
};

// Simple placeholder components (no external dependencies)
const ImageUpload = ({ onAnalysisComplete, loading, setLoading }) => (
  <div style={styles.content}>
    <h2>🩸 Upload Blood Smear Images</h2>
    <p>AI-Powered Medical Imaging Platform</p>
    <input type="file" accept="image/*" multiple style={{margin: '1rem'}} />
    <br />
    <button 
      style={styles.button} 
      disabled={loading}
      onClick={() => {
        setLoading(true);
        setTimeout(() => {
          onAnalysisComplete({
            analysis_id: 'test_123',
            processing_time: 1.5,
            cell_counts: { RBC: 100, WBC: 10, Platelets: 20 }
          });
          setLoading(false);
        }, 1000);
      }}
    >
      {loading ? 'Analyzing...' : 'Analyze Blood Cells'}
    </button>
  </div>
);

const AnalysisResults = ({ results }) => (
  <div style={styles.content}>
    <h2>🔬 Analysis Results</h2>
    {results ? (
      <div>
        <p>Analysis ID: {results.analysis_id}</p>
        <p>Processing Time: {results.processing_time}s</p>
        <p>Cell Counts: RBC: {results.cell_counts?.RBC}, WBC: {results.cell_counts?.WBC}, Platelets: {results.cell_counts?.Platelets}</p>
      </div>
    ) : (
      <p>No analysis results available</p>
    )}
  </div>
);

const Dashboard = () => (
  <div style={styles.content}>
    <h2>📊 Analytics Dashboard</h2>
    <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem'}}>
      <div style={{background: '#667eea', padding: '1rem', borderRadius: '10px', color: 'white'}}>
        <h3>Total Analyses</h3>
        <div style={{fontSize: '2rem', fontWeight: 'bold'}}>156</div>
      </div>
      <div style={{background: '#667eea', padding: '1rem', borderRadius: '10px', color: 'white'}}>
        <h3>Avg Processing Time</h3>
        <div style={{fontSize: '2rem', fontWeight: 'bold'}}>1.8s</div>
      </div>
      <div style={{background: '#667eea', padding: '1rem', borderRadius: '10px', color: 'white'}}>
        <h3>Average Accuracy</h3>
        <div style={{fontSize: '2rem', fontWeight: 'bold'}}>93.2%</div>
      </div>
    </div>
  </div>
);

function App() {
  const [analysisResults, setAnalysisResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('upload');

  const handleAnalysisComplete = (results) => {
    setAnalysisResults(results);
    setActiveTab('results');
  };

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <h1>🩸 Blood Cell Analyzer</h1>
        <p>AI-Powered Medical Imaging Platform</p>
      </header>
      
      <nav style={styles.navTabs} role="navigation">
        <button 
          style={{...styles.button, ...(activeTab === 'upload' ? styles.activeButton : {})}}
          onClick={() => setActiveTab('upload')}
        >
          Upload & Analyze
        </button>
        <button 
          style={{...styles.button, ...(activeTab === 'results' ? styles.activeButton : {})}}
          onClick={() => setActiveTab('results')}
          disabled={!analysisResults}
        >
          Results
        </button>
        <button 
          style={{...styles.button, ...(activeTab === 'dashboard' ? styles.activeButton : {})}}
          onClick={() => setActiveTab('dashboard')}
        >
          Dashboard
        </button>
      </nav>

      <main>
        {activeTab === 'upload' && (
          <ImageUpload 
            onAnalysisComplete={handleAnalysisComplete}
            loading={loading}
            setLoading={setLoading}
          />
        )}
        
        {activeTab === 'results' && (
          <AnalysisResults results={analysisResults} />
        )}
        
        {activeTab === 'dashboard' && (
          <Dashboard />
        )}
      </main>
    </div>
  );
}

export default App;

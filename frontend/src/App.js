import React, { useState } from 'react';
import './App.css';
import ImageUpload from './components/ImageUpload';
import AnalysisResults from './components/AnalysisResults';
import Dashboard from './components/Dashboard';

function App() {
  const [analysisResults, setAnalysisResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('upload');

  const handleAnalysisComplete = (results) => {
    setAnalysisResults(results);
    setActiveTab('results');
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🩸 Blood Cell Analyzer</h1>
        <p>AI-Powered Medical Imaging Platform</p>
      </header>
      
      <nav className="nav-tabs">
        <button 
          className={activeTab === 'upload' ? 'active' : ''}
          onClick={() => setActiveTab('upload')}
        >
          Upload & Analyze
        </button>
        <button 
          className={activeTab === 'results' ? 'active' : ''}
          onClick={() => setActiveTab('results')}
          disabled={!analysisResults}
        >
          Results
        </button>
        <button 
          className={activeTab === 'dashboard' ? 'active' : ''}
          onClick={() => setActiveTab('dashboard')}
        >
          Dashboard
        </button>
      </nav>

      <main className="main-content">
        {activeTab === 'upload' && (
          <ImageUpload 
            onAnalysisComplete={handleAnalysisComplete}
            loading={loading}
            setLoading={setLoading}
          />
        )}
        
        {activeTab === 'results' && analysisResults && (
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
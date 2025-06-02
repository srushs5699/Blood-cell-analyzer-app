
import React, { useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';
import '../styles/AnalysisResults.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const AnalysisResults = ({ results }) => {
  const [activeTab, setActiveTab] = useState('overview');

  if (!results) {
    return <div className="no-results">No analysis results available</div>;
  }

  const cellData = results.cell_counts || {};
  const percentages = results.percentages || {};

  // Chart data for cell counts
  const barChartData = {
    labels: Object.keys(cellData),
    datasets: [
      {
        label: 'Cell Count',
        data: Object.values(cellData),
        backgroundColor: [
          'rgba(255, 99, 132, 0.8)',   // RBC
          'rgba(54, 162, 235, 0.8)',   // WBC
          'rgba(255, 206, 86, 0.8)',   // Platelets
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
        ],
        borderWidth: 2,
      },
    ],
  };

  // Doughnut chart for percentages
  const doughnutData = {
    labels: Object.keys(percentages),
    datasets: [
      {
        data: Object.values(percentages),
        backgroundColor: [
          'rgba(255, 99, 132, 0.8)',
          'rgba(54, 162, 235, 0.8)',
          'rgba(255, 206, 86, 0.8)',
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
        ],
        borderWidth: 2,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Blood Cell Analysis Results',
      },
    },
  };

  const downloadResults = () => {
    const dataStr = JSON.stringify(results, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `blood_analysis_${results.analysis_id || 'results'}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="analysis-results">
      <div className="results-header">
        <h2>🔬 Analysis Results</h2>
        <div className="results-actions">
          <button onClick={downloadResults} className="download-btn">
            📥 Download Results
          </button>
        </div>
      </div>

      <div className="results-summary">
        <div className="summary-card">
          <h3>Total Cells Detected</h3>
          <div className="summary-value">{results.total_cells_detected || 0}</div>
        </div>
        <div className="summary-card">
          <h3>Processing Time</h3>
          <div className="summary-value">{results.processing_time || 0}s</div>
        </div>
        <div className="summary-card">
          <h3>Confidence Score</h3>
          <div className="summary-value">
            {((results.confidence_score || 0) * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      <div className="results-tabs">
        <button
          className={activeTab === 'overview' ? 'active' : ''}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button
          className={activeTab === 'detailed' ? 'active' : ''}
          onClick={() => setActiveTab('detailed')}
        >
          Detailed Analysis
        </button>
        <button
          className={activeTab === 'image' ? 'active' : ''}
          onClick={() => setActiveTab('image')}
        >
          Annotated Image
        </button>
      </div>

      <div className="results-content">
        {activeTab === 'overview' && (
          <div className="overview-tab">
            <div className="charts-container">
              <div className="chart-section">
                <h3>Cell Count Distribution</h3>
                <Bar data={barChartData} options={chartOptions} />
              </div>
              <div className="chart-section">
                <h3>Cell Percentages</h3>
                <Doughnut data={doughnutData} options={chartOptions} />
              </div>
            </div>

            <div className="cell-breakdown">
              <h3>Cell Type Breakdown</h3>
              <div className="cell-types">
                {Object.entries(cellData).map(([cellType, count]) => (
                  <div key={cellType} className="cell-type-card">
                    <div className="cell-type-header">
                      <h4>{cellType}</h4>
                      <span className="cell-count">{count}</span>
                    </div>
                    <div className="percentage-bar">
                      <div
                        className="percentage-fill"
                        style={{
                          width: `${percentages[cellType] || 0}%`,
                          backgroundColor: cellType === 'RBC' ? '#ff6384' :
                                         cellType === 'WBC' ? '#36a2eb' : '#ffce56'
                        }}
                      ></div>
                    </div>
                    <span className="percentage-text">
                      {(percentages[cellType] || 0).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'detailed' && (
          <div className="detailed-tab">
            <div className="detection-details">
              <h3>Detection Details</h3>
              <div className="details-grid">
                <div className="detail-item">
                  <label>Analysis ID:</label>
                  <span>{results.analysis_id || 'N/A'}</span>
                </div>
                <div className="detail-item">
                  <label>Processing Time:</label>
                  <span>{results.processing_time || 0} seconds</span>
                </div>
                <div className="detail-item">
                  <label>Average Confidence:</label>
                  <span>{((results.confidence_score || 0) * 100).toFixed(2)}%</span>
                </div>
                <div className="detail-item">
                  <label>Total Detections:</label>
                  <span>{results.detected_objects?.length || 0}</span>
                </div>
              </div>

              {results.detected_objects && results.detected_objects.length > 0 && (
                <div className="detections-table">
                  <h4>Individual Detections</h4>
                  <table>
                    <thead>
                      <tr>
                        <th>Cell Type</th>
                        <th>Confidence</th>
                        <th>Bounding Box</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.detected_objects.slice(0, 10).map((detection, index) => (
                        <tr key={index}>
                          <td>{detection.class_name}</td>
                          <td>{(detection.confidence * 100).toFixed(1)}%</td>
                          <td>
                            [{detection.bbox.map(coord => Math.round(coord)).join(', ')}]
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {results.detected_objects.length > 10 && (
                    <p className="table-note">
                      Showing first 10 detections of {results.detected_objects.length} total
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'image' && (
          <div className="image-tab">
            <h3>Annotated Image</h3>
            {results.annotated_image ? (
              <div className="annotated-image-container">
                <img
                  src={`data:image/jpeg;base64,${results.annotated_image}`}
                  alt="Annotated blood smear"
                  className="annotated-image"
                />
                <div className="image-legend">
                  <h4>Legend:</h4>
                  <div className="legend-items">
                    <div className="legend-item">
                      <div className="color-box" style={{ backgroundColor: '#ff6384' }}></div>
                      <span>Red Blood Cells (RBC)</span>
                    </div>
                    <div className="legend-item">
                      <div className="color-box" style={{ backgroundColor: '#36a2eb' }}></div>
                      <span>White Blood Cells (WBC)</span>
                    </div>
                    <div className="legend-item">
                      <div className="color-box" style={{ backgroundColor: '#ffce56' }}></div>
                      <span>Platelets</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="no-image">
                <p>No annotated image available</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisResults;

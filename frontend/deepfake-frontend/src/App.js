import React, { useState } from 'react';
import './App.css';

function App() {
  const [fileName, setFileName] = useState('');
  const [prediction, setPrediction] = useState('');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFileName(file.name);
    }
  };

  const handleDetect = () => {
    // Dummy prediction logic
    const result = Math.random() < 0.5 ? 'Real' : 'Fake';
    setPrediction(result);
  };

  return (
    <div className="app">
      <h1 className="title">Deepfake Detection Tool</h1>

      <div className="upload-section">
        <label className="upload-btn">
          Upload Image
          <input type="file" id="fileUpload" onChange={handleFileChange} hidden />
        </label>
        {fileName && <p className="filename">Selected: {fileName}</p>}
      </div>

      <button className="detect-btn" onClick={handleDetect}>
        Detect Deepfake
      </button>

      {prediction && (
        <div className={`result-card ${prediction === 'Real' ? 'real' : 'fake'}`}>
          {prediction === 'Real' ? '✅ Real Video' : '❌ Deepfake Detected'}
        </div>
      )}
    </div>
  );
}

export default App;

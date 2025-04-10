import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://127.0.0.1:8000/predict', formData);
      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Failed to connect to backend.");
    }
    setLoading(false);
  };

  return (
    <div style={styles.container}>
      <h2>🔍 Deepfake Detection</h2>
      <input type="file" onChange={handleFileChange} accept="image/*" />
      <button onClick={handleUpload} style={styles.button}>Detect</button>

      {loading && <p>Loading...</p>}

      {result && (
        <div style={styles.resultBox}>
          <p><strong>Filename:</strong> {result.filename}</p>
          <p><strong>Prediction:</strong> {result.label}</p>
          <p><strong>Confidence:</strong> {(result.prediction * 100).toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: { textAlign: 'center', marginTop: '5rem' },
  button: { margin: '10px', padding: '10px 20px', cursor: 'pointer' },
  resultBox: { marginTop: 20, padding: 15, border: '1px solid #ccc', borderRadius: 10 },
};

export default App;

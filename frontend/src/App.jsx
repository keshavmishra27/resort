import React, { useState } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append("image", file);

    const res = await fetch("http://127.0.0.1:5000/analyze", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    setResult(data);
  }

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input 
          type="file" 
          onChange={e => setFile(e.target.files[0])} 
        />
        <button type="submit">Analyze</button>
      </form>
      {result && (
        <div>
          <p>Total objects: {result.total_objects}</p>
          <p>Class: {result.predicted_class}</p>
        </div>
      )}
    </div>
  );
}


export default App;

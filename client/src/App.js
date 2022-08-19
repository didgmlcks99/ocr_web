import './App.css';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import React from 'react'

function App() {
  const [value, setValue] = React.useState('');
  const [translated, setTranslated] = React.useState('')

  const handleChange = (event) => {
    setValue(event.target.value);
  };

  const buttonClick = (event) => {
    console.log('button clicked!')
    setTranslated('loading...')

    fetch('/translate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text: value,
      }),
    }).then(
        res => res.text()
      ).then(
        translated => {
          setTranslated(translated)
          console.log(translated)
        }
    )

    alert('작성 완료')
  };

  const clearClick = (event) => {
    console.log('clear clicked!')

    setValue('')
    setTranslated('')
  }

  return (
    <div>
      <div>
        <div 
          className="app-header"
          style={{
            textAlign: 'center',
          }}>
            <h2 className="header">한글 라인 브레이커</h2>
        </div>
      </div>

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>

        <div>
          <TextField
            style={{
              width: "500px",
            }}
            id="outlined-multiline-static"
            label="Texts"
            multiline
            rows={15}
            value={value}
            onChange={handleChange}
          />
        </div>
        
        <Button 
          variant="contained"
          onClick={buttonClick}
          style={{
            margin: '20px',
          }}>라인 브레이크</Button>

        <Button 
          variant="contained"
          onClick={clearClick}
          style={{
            margin: '20px',
          }}>삭제</Button>
        
        <div>
          <TextField
            style={{
              width: "500px",
            }}
            id="outlined-multiline-static"
            label="Line Breaked Texts"
            multiline
            rows={15}
            value={translated}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
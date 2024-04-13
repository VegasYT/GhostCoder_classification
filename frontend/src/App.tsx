import React from 'react';
import './css/App.css';
import Banner from './component/Banner';
import Advantages from './component/Advantages';

function App() {
    return (
        <div className="App">
            <div className="main">
                <Banner />
                <Advantages />
            </div>
        </div>
    );
}

export default App;

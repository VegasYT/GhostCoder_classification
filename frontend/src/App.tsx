import React from 'react';
import './css/App.css';
import Banner from './component/Banner';
import Advantages from './component/Advantages';
import Forms from './component/Forms';

function App() {
    return (
        <div className="App">
            <div className="main">
                <Banner />
                <Forms />
                <Advantages />
            </div>
        </div>
    );
}

export default App;

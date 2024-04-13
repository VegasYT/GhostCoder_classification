import React from "react"
import '../css/Preloader.css'


const Preloader:React.FC<{active: boolean}> = (props) => {
    return(
        <div className={`content ${props.active? 'active': ''}`}>
            <div className="loading">
            <p>loading</p>
                <span></span>
            </div>
        </div>
    )
}

export default Preloader;
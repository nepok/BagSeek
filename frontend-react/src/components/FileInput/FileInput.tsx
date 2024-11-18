import React, { useState, useRef, useEffect } from 'react';
import './FileInput.css';

interface FileInputProps {
  onFolderSelect: (folderName: string) => void;
}

const FileInput: React.FC<FileInputProps> = ({ onFolderSelect }) => {
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.setAttribute('webkitdirectory', 'true');
    }
  }, []);

  const handleFolderSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const folderPath = files[0].webkitRelativePath.split('/')[0];
      onFolderSelect(folderPath); // Pass the selected folder name to App
    }
  };

  return (
    <div className="file-input-container">
      <div className="file-input-window">
        <h2>Select a Local Folder</h2>
        <input
          type="file"
          ref={inputRef}
          onChange={handleFolderSelect}
        />
      </div>
    </div>
  );
};

export default FileInput;

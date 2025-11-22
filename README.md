# Time-Series Data Trimmer

An interactive graphical application for exploring, cleaning, and annotating time-series datasets. Built with PyQt6 and pyqtgraph, this tool provides an intuitive interface for visualizing multiple channels of time-series data, marking segments for deletion or annotation, and exporting cleaned datasets with full metadata.

## 🎯 Purpose

This application is designed for researchers, data scientists, and engineers working with high-frequency time-series data (sampled at 120 Hz). It addresses the common challenge of cleaning noisy time-series datasets by providing visual tools to:
- Identify and remove unwanted segments (artifacts, dropouts, etc.)
- Annotate regions of interest with custom labels
- Maintain a complete audit trail of modifications
- Export cleaned data with reproducible annotation metadata

## ✨ Features

### Interactive Visualization
- **Multi-channel plotting**: Display multiple numeric channels simultaneously with synchronized x-axes
- **Dynamic channel selection**: Choose which channels to visualize via checkboxes
- **Real-time updates**: Immediate visual feedback as you interact with the data
- **Shaded regions**: Visual distinction between deleted segments (red) and annotations (blue)

### Segment Management
- **Click-to-mark interface**: Simply click on the plot to set start and end points for segments
- **Flexible selection**: Select any time range for deletion or annotation
- **Visual markers**: Clear blue vertical lines show your current selection

### Data Operations
- **Segment deletion**: Remove unwanted data segments with automatic timeline collapse
- **Time normalization**: Automatically recompute timestamps after deletions (based on 120 Hz sampling)
- **Segment annotation**: Label regions without altering the underlying data
- **Metadata preservation**: Non-numeric columns (metadata) are preserved through all operations

### Annotation System
- **Predefined labels**: Quick access to common labels (blink, dropout, spike, head-movement artefact)
- **Custom labels**: Enter any custom annotation text
- **JSON export**: Save all annotations and deletions for reproducibility
- **JSON import**: Load and reapply previous annotation sessions

### History Management
- **Undo/Redo**: Full undo/redo stack for all deletion and annotation operations
- **State preservation**: Each action is tracked and can be reversed
- **Multiple undo levels**: Go back through multiple operations

### File Operations
- **CSV import/export**: Load and save cleaned datasets in CSV format
- **Annotation persistence**: Save/load annotations and deletions as JSON files
- **Automatic data type detection**: Distinguishes between metadata and signal columns

## 📋 Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
```
PyQt6>=6.0.0
pyqtgraph>=0.12.0
pandas>=1.3.0
numpy>=1.20.0
```

## 🚀 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/aviator1505/Time-Series-Data-Trimmer.git
cd Time-Series-Data-Trimmer
```

### Step 2: Install Dependencies
Using pip:
```bash
pip install PyQt6 pyqtgraph pandas numpy
```

Or create a requirements.txt file with the dependencies listed above and run:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
python main.py
```

## 📖 Usage Guide

### 1. Loading Data
1. Click the **"Load CSV…"** button in the sidebar
2. Select a CSV file with the following structure:
   - Must contain a `normalized_time` column (float, in seconds)
   - Numeric columns will be treated as signal channels
   - Non-numeric columns will be preserved as metadata
3. The application will automatically detect and list all signal channels

### 2. Selecting Channels to Display
- Use the checkboxes in the "Channels" section to show/hide specific signal channels
- By default, the first 5 channels are displayed
- All plots share a common x-axis (time) for easy comparison

### 3. Selecting a Time Segment
1. Click once on any plot to set the **start point** (blue vertical line appears)
2. Click again to set the **end point** (second blue vertical line appears)
3. The status box will show: "Selected segment: X.XXX–Y.YYYs"
4. If both markers are already set, clicking again will reset and start a new selection

### 4. Deleting a Segment
1. Select start and end points as described above
2. Click the **"Delete Segment"** button
3. The selected time range will be:
   - Removed from the dataset
   - Shown as a red shaded region (visible until data refresh)
   - Recorded in the deletions list
4. The timeline will collapse and `normalized_time` will be recomputed
5. Selection markers will be cleared automatically

### 5. Annotating a Segment
1. Select start and end points
2. Choose a label from the dropdown or enter a custom label in the text field
3. Click the **"Annotate Segment"** button
4. The segment will be:
   - Shown as a blue shaded region
   - Preserved in the data (not deleted)
   - Recorded with the label for export

### 6. Undo/Redo Operations
- Click **"Undo"** to reverse the last deletion or annotation
- Click **"Redo"** to reapply an undone operation
- The undo stack persists through multiple operations

### 7. Saving Your Work
**To save cleaned data:**
1. Click **"Save Cleaned CSV…"**
2. Choose a filename and location
3. The current DataFrame (with deletions applied) will be saved

**To save annotations:**
1. Click **"Save Annotations…"**
2. Choose a filename (JSON format recommended)
3. Both annotations and deletions will be saved as JSON

### 8. Loading Previous Annotations
1. Click **"Load Annotations…"**
2. Select a previously saved JSON annotation file
3. The application will:
   - Reset to the original data
   - Reapply all deletions in order
   - Restore all annotations
   - Update the visualization

## 📊 Data Format Requirements

### CSV Structure
Your CSV file must follow these conventions:

**Required Column:**
- `normalized_time`: Float values representing time in seconds (typically computed as sample_index / sample_rate)

**Signal Columns:**
- Any numeric columns (float or integer)
- Will be available for plotting and visualization
- Examples: `channel_1`, `sensor_A`, `acceleration_x`

**Metadata Columns (Optional):**
- Non-numeric columns (strings, categorical data)
- Preserved through all operations but not plotted
- Examples: `subject_id`, `condition`, `notes`

### Example CSV
```csv
normalized_time,channel_1,channel_2,subject_id,condition
0.0000,0.12,0.45,subject_01,baseline
0.0083,0.15,0.48,subject_01,baseline
0.0167,0.13,0.46,subject_01,baseline
...
```

### Annotation JSON Format
When you save annotations, the JSON structure is:
```json
{
  "annotations": [
    {
      "start": 1.234,
      "end": 2.567,
      "label": "blink"
    }
  ],
  "deletions": [
    {
      "start": 5.0,
      "end": 6.5
    }
  ]
}
```

## 🏗️ Architecture Overview

The application is built on four core components:

### 1. DataController
- **Purpose**: Central data management and state control
- **Responsibilities**:
  - Loading/saving CSV files
  - Managing the DataFrame and metadata
  - Tracking deletions and annotations
  - Implementing undo/redo stack
  - Handling data transformations
- **Key Methods**: `load_csv()`, `delete_segment()`, `annotate_segment()`, `undo()`, `redo()`

### 2. SegmentManager
- **Purpose**: Manages temporary marker positions for segment selection
- **Responsibilities**:
  - Tracking start and end marker positions
  - Emitting signals when both markers are set
  - Clearing selections
- **Key Methods**: `set_marker()`, `clear()`

### 3. PlotManager
- **Purpose**: Handles all visualization and plotting logic
- **Responsibilities**:
  - Rendering multi-channel plots with pyqtgraph
  - Drawing marker lines and shaded regions
  - Managing visual feedback for annotations and deletions
  - Updating plots when data changes
- **Key Methods**: `update_plots()`, `draw_marker()`, `update_regions()`

### 4. MainWindow
- **Purpose**: Main GUI controller and user interface
- **Responsibilities**:
  - Constructing the PyQt6 interface
  - Managing sidebar controls and buttons
  - Coordinating interactions between components
  - Handling user events (clicks, button presses)
- **Key Methods**: `on_load_csv()`, `on_delete_segment()`, `on_annotate_segment()`

## ✅ Benefits

1. **Visual and Intuitive**: Direct manipulation of time-series data through graphical interface
2. **Non-destructive Annotations**: Label regions of interest without altering the data
3. **Reproducible**: Save and reload annotations for consistent data cleaning workflows
4. **Flexible**: Works with any numeric time-series data sampled at 120 Hz
5. **Multiple Channels**: View and analyze multiple signals simultaneously
6. **Undo Support**: Experiment freely with full undo/redo capabilities
7. **Metadata Preservation**: Non-numeric columns are maintained through all operations
8. **Audit Trail**: Complete record of all deletions and annotations in JSON format
9. **Lightweight**: Single-file application with minimal dependencies
10. **Open Source**: Fully transparent code that can be customized for specific needs

## ⚠️ Limitations

### Technical Constraints
1. **Fixed Sample Rate**: Hard-coded to 120 Hz (requires code modification for other rates)
2. **Memory Usage**: Entire dataset must fit in memory (not suitable for very large files)
3. **Single File Operation**: Works with one CSV file at a time
4. **No Batch Processing**: Cannot process multiple files automatically
5. **Limited File Formats**: Only CSV input/output (no HDF5, Parquet, etc.)

### Usability Constraints
1. **Manual Selection**: Segment selection requires manual clicking (no automatic detection)
2. **No Zooming Annotations**: Annotation regions are not easily visible when zoomed out
3. **Limited Export**: Cannot export individual channels or filtered data subsets
4. **No Real-time Streaming**: Designed for pre-recorded data only
5. **Platform-Dependent**: GUI behavior may vary slightly across operating systems

### Functional Limitations
1. **No Statistical Analysis**: No built-in tools for computing statistics or features
2. **No Signal Processing**: No filtering, smoothing, or transformation capabilities
3. **No Comparison Views**: Cannot view before/after or compare multiple datasets
4. **No Automatic Artifact Detection**: Requires manual identification of bad segments
5. **Limited Annotation Features**: Cannot edit or delete individual annotations (must undo)

## 🔮 Future Improvements

### High Priority
1. **Configurable Sample Rate**: Allow users to specify sample rate via UI or config file
2. **Annotation Editing**: Add ability to modify or delete specific annotations without undo
3. **Large File Support**: Implement chunked reading for datasets larger than memory
4. **Automatic Artifact Detection**: Add algorithms to detect common artifacts (spikes, dropouts)
5. **Export Options**: Support additional formats (HDF5, Parquet) and selective export

### Medium Priority
6. **Batch Processing Mode**: CLI interface for applying saved annotations to multiple files
7. **Statistics Panel**: Display basic statistics (mean, std, min, max) for selected segments
8. **Enhanced Visualization**: Add spectrogram view, FFT, or other signal processing visualizations
9. **Configuration File**: Support for user preferences and presets
10. **Keyboard Shortcuts**: Add hotkeys for common operations (undo: Ctrl+Z, etc.)

### Low Priority
11. **Plugin System**: Allow custom artifact detectors or processors to be added
12. **Collaborative Annotations**: Support for merging annotations from multiple annotators
13. **Version Control Integration**: Track annotation versions with git-like functionality
14. **Export Report**: Generate PDF summary of cleaning operations and statistics
15. **Unit Tests**: Add comprehensive test coverage for all components

### Architectural Improvements
- Decouple sample rate from code (make it a class attribute or user setting)
- Add configuration file support (YAML or JSON)
- Implement data loading as a plugin system for different formats
- Add logging for debugging and audit purposes
- Create a command-line interface alongside the GUI

## 🐛 Known Issues

- **Plot Click Detection**: In rare cases, clicks near plot edges may not register correctly
- **Memory Management**: Very large datasets (>100 million samples) may cause performance issues
- **Annotation Overlap**: Overlapping annotations are allowed but may be visually confusing

## 🤝 Contributing

Contributions are welcome! Some ways to contribute:
- Report bugs or request features via GitHub Issues
- Submit pull requests for bug fixes or new features
- Improve documentation or add examples
- Test the application with different types of time-series data

## 📝 License

This project is open source. Please check the repository for license information or contact the repository owner.

## 🙏 Acknowledgments

Built with:
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework
- [pyqtgraph](https://www.pyqtgraph.org/) - Scientific plotting library
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [NumPy](https://numpy.org/) - Numerical computing

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the repository maintainer.

---

**Version**: 1.0  
**Last Updated**: November 2025  
**Status**: Active Development

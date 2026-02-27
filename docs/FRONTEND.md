# BagSeek Frontend User Guide

BagSeek has three main views accessible from the top navigation bar: **Search**, **Explore**, and **Map**. This guide walks through each view and common workflows.

---

## Navigation Bar (Header)

The header is always visible at the top of the page.

- **BagSeek** title — click to go to the Search view
- **MAP / EXPLORE / SEARCH** buttons — switch between the three main views

In **Explore** mode, three additional icon buttons appear on the right side:

| Icon | Function |
|------|----------|
| Folder | Select a rosbag to load |
| Canvas | Save / load / delete panel layouts |
| Export | Open the export dialog |

---

## Selecting a Rosbag

Click the **folder icon** in the top-right corner. A dropdown appears listing all available rosbags, each showing:

- Number of MCAP files
- Recording date and time range

Click any rosbag to load it. The rest of the UI updates to reflect the available topics and timestamps from that recording.

> If you have a saved canvas layout whose topics are not available in the newly selected rosbag, the canvas entry will be grayed out.

---

## Search View

The Search view lets you query images across all rosbags using natural language via CLIP embeddings.

### Running a Search

1. Type a query in the text field (e.g. `cow on the field`, `tree stump`).
2. Select a **CLIP model** from the dropdown next to the search field.
3. Click **Search**.

While the search runs, a pipeline status bar shows how many frames pass each filtering stage (MCAP → Topic → Time → Sample → Search).

### Filters (collapsible panel)

Expand the filters section to narrow the search before submitting:

- **Rosbag filter** — restrict to specific rosbags
- **Topic filter** — restrict to specific image topics
- **Time range** — set start/end date and time bounds
- **MCAP range** — per-file slider to limit which file segments are searched
- **Sample rate** — how frequently frames are sampled from the timeline

### Search Results

Results appear as a grid of image cards. Each card shows the matched image, a similarity score, and metadata.

- **Hover** over a card to see a preview in the Rosbag Overview section below
- **Click the arrow icon** on a card to open that frame in the Explore view (automatically jumps to the correct rosbag and timestamp)
- **Right-click** a card to download the image or initiate an export
- **Click "Search with Image"** on a card to use that image as the next query

### Rosbag Overview (below results)

This section organizes all results by rosbag → model → topic. Each row displays a **heatbar** showing where in the timeline matches occur.

- **Hover** over the heatbar to preview the image at that position
- **Drag** across the heatbar to select a time range, then right-click to export that section
- **Click the arrow icon** on a row to open that rosbag at the matching position in Explore

---

## Explore View

The Explore view shows sensor data as a resizable grid of panels, synchronized to a shared timeline.

### Setting Up Panels

Each panel is empty by default. To assign a data stream:

1. Click the **⋮ menu** in the top-right corner of any panel.
2. Select **Choose Topic** and pick a topic from the dropdown.

The panel will load and display the data for the current timestamp.

**Splitting panels:**

- From the ⋮ menu, select **Split Horizontally** (side by side) or **Split Vertically** (stacked).
- **Drag the divider** between panels to resize them.
- Select **Delete Panel** to remove a panel.

### Supported Data Types

| Data type | Rendering |
|-----------|-----------|
| Images | JPEG/PNG image |
| Point clouds | Interactive 3D view (Three.js) |
| GPS / Position | Interactive map with path heatmap |
| IMU | 3D orientation visualization |
| Odometry | Pose/transform display |

### Navigating the Timeline

The **TimestampPlayer** bar at the bottom controls the current timestamp:

- **Drag the slider** to scrub through the recording.
- Click **Play/Pause** to start/stop automatic playback.
- Use the **speed selector** to change playback rate (0.5×, 1×, 2×).
- Toggle **ROS** / **TOD** to switch between nanosecond and time-of-day display.

If a search was performed, **search marks** appear as colored ticks on the slider — brighter ticks indicate higher similarity. Click any tick to jump to that frame.

MCAP file boundaries appear as dividers on the slider, so you can see where one file segment ends and the next begins.

### Saving and Loading Layouts

Click the **canvas icon** in the header to manage layouts:

- Click **+ Add**, type a name, and press Enter to save the current panel arrangement.
- Click a saved canvas name to restore it.
- Click the delete icon next to a name to remove it.

Saved layouts store which topics are assigned to which panels and their sizes.

---

## Map View

The Map view shows the geographic coverage of all rosbags on an OpenStreetMap tile layer with a heatmap overlay.

### Exploring Coverage

- **Toggle rosbag layers** using the layer controls to show/hide individual recordings.
- Each rosbag is color-coded and shows its GPS path and density heatmap.
- MCAP segment boundaries are visualized within each layer.

### Polygon Filtering

You can draw a geographic region to filter which MCAP files to work with:

1. Click **Draw Mode** to enable polygon drawing.
2. Click on the map to add vertices.
3. When your cursor is near the first point, it snaps closed — click to finish the polygon.
4. Right-click anywhere to reset the drawing.

After drawing:

- The panel shows which MCAP files overlap with the polygon and how many frames each contains.
- Click **Open in Explore** to switch to the Explore view with only those MCAP files highlighted on the timeline slider.

---

## Exporting Data

The export dialog can be opened from:

- The **export icon** in the header (Explore mode)
- **Right-click → Export** on a search result card
- **Drag-select on a heatbar** in the Rosbag Overview, then right-click
- The **Open in Explore** button from the Map view (pre-selects overlapping MCAPs)

### Export Dialog

**1. Select Rosbags**
Check the rosbags you want to include. Use **All** to toggle all at once.

**2. Select Topics**
Switch between **By Topic** and **By Type** tabs:
- Expand the tree to find specific topics.
- Use **Select All** / **Clear All** for bulk selection.
- Save frequently used selections as **Topic Presets** using the preset dropdown.

**3. Set Time Range**
- The overall time bounds are shown automatically.
- Use the **MCAP Range Filter** to set per-file start/end offsets with sliders.
- The heatbar shows data density — the current Explore timestamp is marked.

**4. Configure Output Name**
Optionally include in the output filename:
- Original rosbag name
- MCAP range info
- Part number
- Custom text prefix

**5. Export**
Click **Export**. A progress bar appears. When complete, a download link is provided.

---

## Tips

- The URL updates as you navigate, so you can **bookmark or share** a specific rosbag + timestamp + panel layout.
- Click the **? help icon** that appears in various sections for context-specific guidance.
- Notifications (errors, loading states) appear as a snackbar in the bottom-left corner.

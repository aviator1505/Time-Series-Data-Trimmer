# PyInstaller spec for the Time-Series Data Trimmer.
# Build locally with:  pyinstaller tsdt.spec
# CI builds per-OS artifacts via .github/workflows/build.yml.

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        # pyqtgraph loads its OpenGL items dynamically
        "pyqtgraph.opengl",
        "OpenGL",
        # pandas Arrow IO used by .tsdt session bundles
        "pyarrow",
        "pyarrow.feather",
        # scipy signal path used by the filter engine
        "scipy.signal",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "pytest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="tsdt",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="TimeSeriesDataTrimmer",
)

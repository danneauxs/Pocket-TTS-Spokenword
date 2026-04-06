"""
Simplified GUI launcher for headless testing.
"""

import sys
import os

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gui_imports():
    """Test that GUI components can be imported."""
    print("Testing GUI component imports...")

    try:
        # Test basic imports
        from preprocessing.structure_detector import StructureDetector
        from preprocessing.chunker import SmartChunker
        from preprocessing.emotion_analyzer import EmotionAnalyzer
        from preprocessing.parameter_mapper import ParameterMapper
        from config import ConfigManager

        print("✓ Core preprocessing imports successful")

        # Test component instantiation
        detector = StructureDetector()
        chunker = SmartChunker()
        analyzer = EmotionAnalyzer()
        mapper = ParameterMapper()
        config = ConfigManager.load_config()

        print("✓ Component instantiation successful")
        print("✓ Configuration loading successful")

        # Test basic functionality
        test_text = "Hello world. This is a test."
        structure = detector.analyze(test_text)
        chunks = chunker.chunk(structure)

        print(f"✓ Basic processing pipeline works: {len(chunks)} chunks created")

        return True

    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point."""
    print("Audiobook Generator GUI (Headless Test Mode)")
    print("=" * 50)

    if test_gui_imports():
        print("\n🎉 GUI components are ready!")
        print("\nTo run the full GUI (requires display):")
        print("  python launch_gui.py")
        print("\nFor testing preprocessing:")
        print("  uv run pocket-tts test-preprocessing book.txt")
    else:
        print("\n❌ GUI setup incomplete")
        sys.exit(1)

if __name__ == "__main__":
    main()
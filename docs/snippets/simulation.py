# --- Test Scenario Simulation ---

# 1. Define the input and what we expect to find
TEST_SCREENSHOT = "data/datasource/test_screenshot_login.png"
TARGET_ELEMENT = "login_button"
MIN_CONFIDENCE = 0.80

print(f"Searching for '{TARGET_ELEMENT}' in {TEST_SCREENSHOT}...")

# 2. Run the YOLO detection
found_elements = find_element_coordinates(TEST_SCREENSHOT, TARGET_ELEMENT, MIN_CONFIDENCE)

# 3. Process the Result for openQA
if found_elements:
    # We'll take the first (and usually most confident) detection
    best_match = found_elements[0]
    
    # Calculate the center point for the click action
    x1, y1, x2, y2 = best_match['x1'], best_match['y1'], best_match['x2'], best_match['y2']
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    print("\n1: Element Found!")
    print(f"  Element: **{best_match['element']}** (Confidence: {best_match['confidence']})")
    print(f"  Bounding Box (x1, y1, x2, y2): ({x1}, {y1}, {x2}, {y2})")
    
    # --- The output to the openQA Test Framework ---
    # This is the key output replacing the 'needle' match
    print(f"  **ACTION COORDINATES (Click Center):** ({center_x}, {center_y})")
    # Simulate a click action in the test environment (e.g., using a test utility)
    # openqa_utils.mouse_click(center_x, center_y) or send_key()
    
    # If the confidence is very high, we can pass the test step.
    
else:
    # --- Handling the Anomalous/Unknown State (from previous step) ---
    print("\n0: Element NOT Found or Confidence Too Low!")
    print(f"  This could indicate a test failure, a UI change, or an ANOMALOUS screen state.")
    # openqa_utils.fail_test("Target element missing. Review screenshot.")
# end

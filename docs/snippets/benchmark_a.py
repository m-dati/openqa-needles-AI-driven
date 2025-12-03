import time

start_time = time.perf_counter()
for image_path, target_element in test_list:
    find_element_coordinates(image_path, target_element)
end_time = time.perf_counter()

total_time = end_time - start_time
avg_ai_time = total_time / len(test_list)

print(f"YOLOv8 Average Inference Time: {avg_ai_time * 1000:.2f} ms per image")

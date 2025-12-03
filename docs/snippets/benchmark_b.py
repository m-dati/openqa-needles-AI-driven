# Pseudo-code for openQA comparison
start_time_old = time.perf_counter()
for image_path, needle_data in test_list:
    openqa_match_function(image_path, needle_data) 
end_time_old = time.perf_counter()

avg_needle_time = (end_time_old - start_time_old) / len(test_list)
print(f"Needles Average Matching Time: {avg_needle_time * 1000:.2f} ms per image")

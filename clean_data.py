
cleaned_data = ""
cleaned_p_count = 0
cleaned_n_count = 0

with open('./dataset/crx.data.txt', 'r') as f:
	data = f.readlines()
	for i, row in enumerate(data):
		# Check for '?' value in each row (indicates missing)
		if '?' not in row:
			cleaned_data += row
			if '+' in row:
				cleaned_p_count += 1
			elif '-' in row:
				cleaned_n_count += 1

	print(cleaned_data)

with open('./dataset/crx_clean.data.txt', 'w') as f:
	f.write(cleaned_data)

with open('./dataset/crx_clean.names.txt', 'w') as f:
	f.write("Class Distribution\n")
	f.write("+ Classes: %d\n" %cleaned_p_count)
	f.write("- Classes: %d\n" %cleaned_n_count)
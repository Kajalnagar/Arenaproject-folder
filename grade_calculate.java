# Student Grade Calculator

def calculate_grade(marks):
    if marks >= 90 and marks <= 100:
        return "A", "Excellent Work! "
    elif marks >= 80:
        return "B", "Very Good! Keep it up! "
    elif marks >= 70:
        return "C", "Good job! Keep improving!"
    elif marks >= 60:
        return "D", "You passed! Work a little harder."
    else:
        return "F", "Don't give up! Try again."

# Get student name
name = input("Enter student name: ")

# Input validation using while loop
while True:
    marks = int(input("Enter marks (0-100): "))

    if 0 <= marks <= 100:
        break
    else:
        print("Invalid input! Marks must be between 0 and 100.")

# Calculate grade
grade, message = calculate_grade(marks)

# Display result
print("\ RESULT FOR", name.upper())
print("Marks:", marks, "/100")
print("Grade:", grade)
print("Message:", message)

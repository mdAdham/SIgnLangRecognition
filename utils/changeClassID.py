import csv

# Name der CSV-Datei
csv_file = "keypoint.csv"

# Index der Spalte, deren erste Ziffer geändert werden soll (0-basiert)
spalten_index = 0

# Öffne die CSV-Datei im Lese- und Schreibmodus
with open(csv_file, 'r', newline='') as file:
    reader = csv.reader(file)
    rows = list(reader)

# Iteriere über die Zeilen der CSV-Datei und ändere die erste Ziffer in der ausgewählten Spalte
for row in rows:
    if len(row[spalten_index]) > 0:
        first_digit = int(row[spalten_index][0])
        new_value = "10" + str(row[spalten_index])[1:]
        row[spalten_index] = new_value

# Speichere die geänderten Daten zurück in die CSV-Datei
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)

print("Die erste Ziffer in der ausgewählten Spalte wurde erfolgreich geändert.")

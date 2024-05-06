def total_euro(working_hours, eur_per_hour):
    return working_hours * eur_per_hour


working_hours = float(input("Radni sati: "))
eur_per_hour = float(input("eura/h: "))
print("Ukupno: {0} eura".format(total_euro(working_hours, eur_per_hour)))
5
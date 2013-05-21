"""
Functions for processing DEFRA's Home Energy Survey dataset (once
loaded into an HDF5 file).

More info on DEFRA's Household Electricity Survey:
http://randd.defra.gov.uk/Default.aspx?Menu=Menu&Module=More&Location=None&Completed=0&ProjectID=17359
"""

def load_appliance_labels(filename='/data/HES/CSVdata/appliance_list.csv'):
    f = open(filename, 'r')
    lines = f.readlines()[1:]
    f.close()

    # 0: ApplianceGroupID
    # 1: Appliance
    # 2: Group
    # 3: Index
    # 4: ApplianceName
    # 5: TypeOfData
    # 6: IsUsed
    # 7: WaterUsingProduct
    # e.g.:
    # '0,255,15,15,"Inside T degrees 4","Temperature;dgrees C;0,1||",1,0\r\n'

    labels = {}
    for line in lines:
        line = line.strip()
        split_line = line.split(',')
        
        appliance_id = int(split_line[1])
        label = split_line[4].strip('"')

        labels[appliance_id] = label

    return labels
        

def plot_whole_house(df, labels, axes):
    for appliance_id in df:
        series = df[appliance_id]
        axes.plot(series.index, series, label=labels[appliance_id])

    return axes

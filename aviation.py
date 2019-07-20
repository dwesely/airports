'''
Do not use output of this script for navigation.

'''

import zipfile
import csv
import pandas as pd
from io import StringIO
import datetime
import nltk
import numpy as np


INSPECTION_THRESHOLD = 365*5 # days, inspections before this age probably indicate a closed/outdated airport

class Airport:
    def __init__(self,
                 id='',
                 name='',
                 lat=0,
                 lon=0,
                 lat_dms=[],
                 lon_dms=[],
                 lat_dms_string='',
                 lon_dms_string='',
                 city='',
                 state='',
                 icao='',
                 start_date='',
                 end_date='',
                 status='',
                 source=''):
        self.id = id
        self.name = name.upper()
        self.lat = lat
        self.lon = lon
        if (lat or lon) and not lat_dms:
            lat_dms, lon_dms = ll_decimal_to_dms(lat,lon)
        if (lat_dms and lon_dms) and not lat_dms_string:
            lat_dms_string, lon_dms_string = ll_dms_to_string(lat_dms, lon_dms)
        self.lat_dms = lat_dms
        self.lon_dms = lon_dms
        self.lat_dms_string = lat_dms_string
        self.lon_dms_string = lon_dms_string
        self.city = city.split(',')[0].upper()
        self.state = state
        self.icao = icao
        self.start_date = start_date
        self.end_date = end_date
        self.status = status
        self.source = source

def get_dms(decimal_degrees):
    decimal_degrees = abs(decimal_degrees)
    degrees = np.floor(decimal_degrees)
    remainder = decimal_degrees - degrees
    minutes = np.floor(remainder*60.0)
    remainder = remainder - (minutes/60.0)
    seconds = round(remainder*60.0*60.0, 4)
    return degrees, minutes, seconds

def ll_decimal_to_dms(lat,lon):
    '''
    Convert decimal latitude/longitude to degrees minutes seconds
    '''
    if lat < 0:
        lat_hemisphere = 'S'
    else:
        lat_hemisphere = 'N'
    lat_deg, lat_min, lat_sec = get_dms(lat)

    if lon < 0:
        lon_hemisphere = 'W'
    else:
        lon_hemisphere = 'E'

    lon_deg, lon_min, lon_sec = get_dms(lon)

    lat_dms = [lat_hemisphere, lat_deg, lat_min, lat_sec]
    lon_dms = [lon_hemisphere, lon_deg, lon_min, lon_sec]

    return lat_dms, lon_dms

def ll_dms_to_decimal(lat_dms,lon_dms):
    '''
    Convert degrees minutes seconds to decimal latitude/longitude
    '''
    return lat,lon

def ll_dms_to_string(lat_dms, lon_dms):
    '''

    '''
    lat_dms_string = ''.join([lat_dms[0],
                              str(int(lat_dms[1])).zfill(2),
                              str(int(lat_dms[2])).zfill(2),
                              str(int(lat_dms[3])).zfill(2)])

    lon_dms_string = ''.join([lon_dms[0],
                              str(int(lon_dms[1])).zfill(3),
                              str(int(lon_dms[2])).zfill(2),
                              str(int(lon_dms[3])).zfill(2)])
    return lat_dms_string, lon_dms_string

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    # Function source:
    # https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    # https://creativecommons.org/licenses/by-sa/3.0/

    # make the first value the same length as the second
    if type(lon2) is list and type(lon1) is not list:
        lon1 = [lon1] * len(lon2)
        lat1 = [lat1] * len(lon2)

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def airports_to_df(airports):
    header = ['id','name','lat','lon','lat_dms_string','lon_dms_string','lat_hemisphere','lat_deg','lat_min','lat_sec','lon_hemisphere','lon_deg','lon_min','lon_sec','city','state','icao','start_date','end_date','status','source']
    data = [[a.id, a.name, a.lat, a.lon,
             a.lat_dms_string, a.lon_dms_string,
             a.lat_dms[0], int(a.lat_dms[1]), int(a.lat_dms[2]), int(a.lat_dms[3]),
             a.lon_dms[0], int(a.lon_dms[1]), int(a.lon_dms[2]), int(a.lon_dms[3]),
             a.city, a.state, a.icao, a.start_date, a.end_date, a.status, a.source] for a in airports]
    airport_df = pd.DataFrame.from_records(data, columns=header)
    return airport_df


def get_usgs_airport_list(flat_file):
    airport_df = get_usgs_airport_df(flat_file)
    airports = []
    for index, row in airport_df.iterrows():
        # if pd.isna(row['LAT_DEGREES']):
        #     print('Skipping unknown airport:')
        #     print(row)
        #     continue

        airport = Airport(id=row['FAA_AIRPOR'],
                          name=row['NAME'],
                          lat=row['Y'],
                          lon=row['X'],
                          source='usgs'
                          )
        airports.append(airport)
    return airports

def get_usgs_airport_df(flat_file):
    """Read USGS AirportPoint list and return airport details

    Prepped this data by opening the shapefile in QGIS, saving to a csv with GEOMETRY=AS_XY

    Source data:
    https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Tran/Shape/
    """
    # TODO: Download an updated file from https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=288&DB_Short_Name=Aviation%20Support%20Tables
    # TODO: Allow closed or open runways to be returned

    with open(flat_file, 'r') as csvfile:
        file_contents = StringIO(csvfile.read())
        df = pd.read_csv(file_contents)
        us_airports = df.loc[(df.GEODB_SUB != 'Runway')]
        return us_airports

def get_bts_airport_list(archive):
    airport_df = get_bts_airport_df(archive)
    airports = []
    for index, row in airport_df.iterrows():
        if pd.isna(row['LAT_DEGREES']):
            print('Skipping unknown airport:')
            print(row)
            continue

        lat_dms = [row['LAT_HEMISPHERE'],
                   row['LAT_DEGREES'],
                   row['LAT_MINUTES'],
                   row['LAT_SECONDS']]

        lon_dms = [row['LON_HEMISPHERE'],
                   row['LON_DEGREES'],
                   row['LON_MINUTES'],
                   row['LON_SECONDS']]

        status = 'O'
        if row['AIRPORT_IS_CLOSED'] == 1:
            status = 'C'
        start_date = ''
        if not pd.isna(row['AIRPORT_START_DATE']):
            start_date = datetime.datetime.strptime(row['AIRPORT_START_DATE'], '%Y-%m-%d')

        end_date = ''
        if not pd.isna(row['AIRPORT_THRU_DATE']):
            end_date = datetime.datetime.strptime(row['AIRPORT_THRU_DATE'], '%Y-%m-%d')
        airport = Airport(id=row['AIRPORT'],
                          name=row['DISPLAY_AIRPORT_NAME'],
                          lat=row['LATITUDE'],
                          lon=row['LONGITUDE'],
                          lat_dms=lat_dms,
                          lon_dms=lon_dms,
                          city=row['DISPLAY_AIRPORT_CITY_NAME_FULL'],
                          state=row['AIRPORT_STATE_CODE'],
                          icao='',
                          start_date=start_date,
                          end_date=end_date,
                          status=status,
                          source='bts'
                          )
        airports.append(airport)
    return airports

def get_bts_airport_df(archive):
    """Read BTS Master Coordinates list and return airport details
    Source data:
    https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=288&DB_Short_Name=Aviation%20Support%20Tables
    """
    # TODO: Download an updated file from https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=288&DB_Short_Name=Aviation%20Support%20Tables
    # TODO: Allow closed or open runways to be returned
    zf = zipfile.ZipFile(archive)
    contained_files = zf.namelist()

    with zf.open(contained_files[0], 'r') as csvfile:
        file_contents = StringIO(csvfile.read().decode('utf-8'))
        df = pd.read_csv(file_contents)
        us_airports = df.loc[(df.AIRPORT_IS_LATEST == 1) &
                             (df.AIRPORT_COUNTRY_CODE_ISO == 'US')   ]
        us_airports = us_airports[['AIRPORT',
                                   'DISPLAY_AIRPORT_NAME',
                                   'DISPLAY_AIRPORT_CITY_NAME_FULL',
                                   'AIRPORT_STATE_CODE',
                                   'LATITUDE',
                                   'LAT_HEMISPHERE',
                                   'LAT_DEGREES',
                                   'LAT_MINUTES',
                                   'LAT_SECONDS',
                                   'LONGITUDE',
                                   'LON_HEMISPHERE',
                                   'LON_DEGREES',
                                   'LON_MINUTES',
                                   'LON_SECONDS',
                                   'AIRPORT_IS_CLOSED',
                                   'AIRPORT_START_DATE',
                                   'AIRPORT_THRU_DATE'
                                   ]]
        return us_airports

def get_ourairports_airports_df(archive):
    '''
    Parse ourairports airport locations

    source data:
    http://ourairports.com/data/
    '''


def get_abandoned_airports_df(archive):
    '''
    Parse abandoned airfields airport locations

    source data:
    http://www.airfields-freeman.com/
    '''

def get_nfdc_airport_list(archive):
    airport_df = get_nfdc_airport_df(archive)
    airports = []
    for index, row in airport_df.iterrows():
        # us_airports = df[['LOCATION IDENTIFIER',
        #                   'LANDING FACILITY TYPE',
        #                   'BOUNDARY ARTCC IDENTIFIER',
        #                   'RESPONSIBLE ARTCC IDENTIFIER',
        #                   'AIR TRAFFIC CONTROL TOWER LOCATED ON AIRPORT',
        #                   'LENS COLOR OF OPERABLE BEACON LOCATED ON THE AIRPORT',
        #                   'ICAO IDENTIFIER'
        #                   ]]

        lat_formatted = row['AIRPORT REFERENCE POINT LATITUDE (FORMATTED)']

        lat_dms = [lat_formatted[-1],
                   float(lat_formatted[:2]),
                   float(lat_formatted[3:5]),
                   float(lat_formatted[6:13])]
        lat = row['AIRPORT REFERENCE POINT LATITUDE (SECONDS)']
        lat = float(lat[0:-1])/3600.0
        if lat_dms[0] == 'S':
            lat = lat*(-1)

        lon_formatted = row['AIRPORT REFERENCE POINT LONGITUDE (FORMATTED)']

        lon_dms = [lon_formatted[-1],
                   float(lon_formatted[:3]),
                   float(lon_formatted[4:6]),
                   float(lon_formatted[7:14])]
        lon = row['AIRPORT REFERENCE POINT LONGITUDE (SECONDS)']
        lon = float(lon[0:-1])/3600.0
        if lon_dms[0] == 'W':
            lon = lon*(-1)

        status = row['AIRPORT STATUS CODE']

        start_date = ''
        if not pd.isna(row['AIRPORT ACTIVATION DATE (MM/YYYY)']):
            start_date = datetime.datetime.strptime(row['AIRPORT ACTIVATION DATE (MM/YYYY)'], '%m/%Y')

        end_date = ''
        if not pd.isna(row['LAST PHYSICAL INSPECTION DATE (MMDDYYYY)']):
            inspection_date = str(int(row['LAST PHYSICAL INSPECTION DATE (MMDDYYYY)'])).zfill(8)
            end_date = datetime.datetime.strptime(inspection_date, '%m%d%Y')
            if datetime.datetime.now() - end_date < datetime.timedelta(days = INSPECTION_THRESHOLD):
                # This aiport was inspected resently, probably not a closure time?
                end_date = ''

        airport = Airport(id=row['LOCATION IDENTIFIER'],
                          name=row['OFFICIAL FACILITY NAME'],
                          lat=lat,
                          lon=lon,
                          lat_dms=lat_dms,
                          lon_dms=lon_dms,
                          city=row['ASSOCIATED CITY NAME'],
                          state=row['ASSOCIATED STATE POST OFFICE CODE'],
                          icao=row['ICAO IDENTIFIER'],
                          start_date=start_date,
                          end_date=end_date,
                          status=status,
                          source='nfdc'
                          )
        airports.append(airport)
    return airports

def get_nfdc_airport_df(archive):
    """Read NFDC airport list and return: Country, State, Airport name, Lat, Lon, Operational

    source data:
    https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/NASR_Subscription/
    """
    airport_list = []
    nfdc_filename = 'nfdc_airports.csv'
    nfdc_file = open(nfdc_filename, 'w')
    nfdc_header = '"' + '","'.join(
        ["RECORD TYPE INDICATOR".replace("'", ''), "LANDING FACILITY SITE NUMBER".replace("'", ''),
         "LANDING FACILITY TYPE".replace("'", ''), "LOCATION IDENTIFIER".replace("'", ''),
         "INFORMATION EFFECTIVE DATE (MM/DD/YYYY)".replace("'", ''), "FAA REGION CODE".replace("'", ''),
         "FAA DISTRICT OR FIELD OFFICE CODE".replace("'", ''), "ASSOCIATED STATE POST OFFICE CODE".replace("'", ''),
         "ASSOCIATED STATE NAME".replace("'", ''), "ASSOCIATED COUNTY (OR PARISH) NAME".replace("'", ''),
         "ASSOCIATED COUNTY'S STATE (POST OFFICE CODE)".replace("'", ''), "ASSOCIATED CITY NAME".replace("'", ''),
         "OFFICIAL FACILITY NAME".replace("'", ''), "AIRPORT OWNERSHIP TYPE".replace("'", ''),
         "FACILITY USE".replace("'", ''), "FACILITY OWNER'S NAME".replace("'", ''), "OWNER'S ADDRESS".replace("'", ''),
         "OWNER'S CITY, STATE AND ZIP CODE".replace("'", ''), "FACILITY MANAGER'S NAME".replace("'", ''),
         "MANAGER'S ADDRESS".replace("'", ''), "MANAGER'S CITY, STATE AND ZIP CODE".replace("'", ''),
         "AIRPORT REFERENCE POINT LATITUDE (FORMATTED)".replace("'", ''),
         "AIRPORT REFERENCE POINT LATITUDE (SECONDS)".replace("'", ''),
         "AIRPORT REFERENCE POINT LONGITUDE (FORMATTED)".replace("'", ''),
         "AIRPORT REFERENCE POINT LONGITUDE (SECONDS)".replace("'", ''),
         "AIRPORT REFERENCE POINT DETERMINATION METHOD".replace("'", ''),
         "AIRPORT ELEVATION DETERMINATION METHOD".replace("'", ''), "MAGNETIC VARIATION AND DIRECTION".replace("'", ''),
         "MAGNETIC VARIATION EPOCH YEAR".replace("'", ''),
         "AERONAUTICAL SECTIONAL CHART ON WHICH FACILITY".replace("'", ''),
         "DISTANCE FROM CENTRAL BUSINESS DISTRICT OF".replace("'", ''),
         "DIRECTION OF AIRPORT FROM CENTRAL BUSINESS".replace("'", ''), "BOUNDARY ARTCC IDENTIFIER".replace("'", ''),
         "BOUNDARY ARTCC (FAA) COMPUTER IDENTIFIER".replace("'", ''), "BOUNDARY ARTCC NAME".replace("'", ''),
         "RESPONSIBLE ARTCC IDENTIFIER".replace("'", ''),
         "RESPONSIBLE ARTCC (FAA) COMPUTER IDENTIFIER".replace("'", ''), "RESPONSIBLE ARTCC NAME".replace("'", ''),
         "TIE-IN FSS PHYSICALLY LOCATED ON FACILITY".replace("'", ''),
         "TIE-IN FLIGHT SERVICE STATION (FSS) IDENTIFIER".replace("'", ''), "TIE-IN FSS NAME".replace("'", ''),
         "LOCAL PHONE NUMBER FROM AIRPORT TO FSS".replace("'", ''),
         "TOLL FREE PHONE NUMBER FROM AIRPORT TO FSS".replace("'", ''), "ALTERNATE FSS IDENTIFIER".replace("'", ''),
         "ALTERNATE FSS NAME".replace("'", ''), "TOLL FREE PHONE NUMBER FROM AIRPORT TO".replace("'", ''),
         "IDENTIFIER OF THE FACILITY RESPONSIBLE FOR".replace("'", ''),
         "AVAILABILITY OF NOTAM 'D' SERVICE AT AIRPORT".replace("'", ''),
         "AIRPORT ACTIVATION DATE (MM/YYYY)".replace("'", ''), "AIRPORT STATUS CODE".replace("'", ''),
         "AIRPORT ARFF CERTIFICATION TYPE AND DATE".replace("'", ''), "NPIAS/FEDERAL AGREEMENTS CODE".replace("'", ''),
         "AIRPORT AIRSPACE ANALYSIS DETERMINATION".replace("'", ''),
         "FACILITY HAS BEEN DESIGNATED BY THE U.S. TREASURY".replace("'", ''),
         "FACILITY HAS BEEN DESIGNATED BY THE U.S. TREASURY".replace("'", ''),
         "FACILITY HAS MILITARY/CIVIL JOINT USE AGREEMENT".replace("'", ''),
         "AIRPORT HAS ENTERED INTO AN AGREEMENT THAT".replace("'", ''), "AIRPORT INSPECTION METHOD".replace("'", ''),
         "AGENCY/GROUP PERFORMING PHYSICAL INSPECTION".replace("'", ''),
         "LAST PHYSICAL INSPECTION DATE (MMDDYYYY)".replace("'", ''),
         "LAST DATE INFORMATION REQUEST WAS COMPLETED".replace("'", ''),
         "FUEL TYPES AVAILABLE FOR PUBLIC USE AT THE".replace("'", ''),
         "AIRFRAME REPAIR SERVICE AVAILABILITY/TYPE".replace("'", ''),
         "POWER PLANT (ENGINE) REPAIR AVAILABILITY/TYPE".replace("'", ''),
         "TYPE OF BOTTLED OXYGEN AVAILABLE (VALUE REPRESENTS".replace("'", ''),
         "TYPE OF BULK OXYGEN AVAILABLE (VALUE REPRESENTS".replace("'", ''),
         "AIRPORT LIGHTING SCHEDULE".replace("'", ''), "BEACON LIGHTING SCHEDULE".replace("'", ''),
         "AIR TRAFFIC CONTROL TOWER LOCATED ON AIRPORT".replace("'", ''),
         "UNICOM FREQUENCY AVAILABLE AT THE AIRPORT".replace("'", ''),
         "COMMON TRAFFIC ADVISORY FREQUENCY (CTAF)".replace("'", ''),
         "SEGMENTED CIRCLE AIRPORT MARKER SYSTEM ON THE AIRPORT".replace("'", ''),
         "LENS COLOR OF OPERABLE BEACON LOCATED ON THE AIRPORT".replace("'", ''),
         "LANDING FEE CHARGED TO NON-COMMERCIAL USERS OF".replace("'", ''),
         "A Y IN THIS FIELD INDICATES THAT THE LANDING".replace("'", ''),
         "12-MONTH ENDING DATE ON WHICH ANNUAL OPERATIONS DATA".replace("'", ''),
         "AIRPORT POSITION SOURCE".replace("'", ''), "AIRPORT POSITION SOURCE DATE (MM/DD/YYYY)".replace("'", ''),
         "AIRPORT ELEVATION SOURCE".replace("'", ''), "AIRPORT ELEVATION SOURCE DATE (MM/DD/YYYY)".replace("'", ''),
         "CONTRACT FUEL AVAILABLE".replace("'", ''), "TRANSIENT STORAGE FACILITIES".replace("'", ''),
         "OTHER AIRPORT SERVICES AVAILABLE".replace("'", ''), "WIND INDICATOR".replace("'", ''),
         "ICAO IDENTIFIER".replace("'", ''), "AIRPORT RECORD FILLER (BLANK)".replace("'", '')]) + '"'
    nfdc_file.write(nfdc_header)

    i = 0
    zf = zipfile.ZipFile(archive)

    with zf.open('APT.txt', 'r') as aptfile:
        line = aptfile.readline()
        while len(line) > 0:
            try:
                line = line.decode('utf-8').replace(',', ';')
                i = i + 1
                if line[0:3] == 'APT':
                    latsec = line[538:550].strip()
                    lonsec = line[565:577].strip()
                    NS = 1
                    if latsec[-1] == 'S':
                        NS = -1
                    EW = 1
                    if lonsec[-1] == 'W':
                        EW = -1
                    airport_list.append({'airport': line[133:183].strip(),
                                         'lat': NS * float(latsec[0:-1]) / 3600,
                                         'lon': EW * float(lonsec[0:-1]) / 3600,
                                         'link': 'nfdc',
                                         'state': line[48:50].strip(),
                                         'city': line[93:133].strip(),
                                         'start': line[31:41].strip(),
                                         'id': line[27:31].strip()})
                    if len(line[48:50].strip()) > 0:
                        nfdc_line = '\n"' + '","'.join(
                            [line[0:3].strip(), line[3:14].strip(), line[14:27].strip(), line[27:31].strip(),
                             line[31:41].strip(), line[41:44].strip(), line[44:48].strip(), line[48:50].strip(),
                             line[50:70].strip(), line[70:91].strip(), line[91:93].strip(), line[93:133].strip(),
                             line[133:183].strip(), line[183:185].strip(), line[185:187].strip(), line[187:222].strip(),
                             line[222:294].strip(), line[294:339].strip(), line[355:390].strip(), line[390:462].strip(),
                             line[462:507].strip(), line[523:538].strip(), line[538:550].strip(), line[550:565].strip(),
                             line[565:577].strip(), line[577:578].strip(), line[585:586].strip(), line[586:589].strip(),
                             line[589:593].strip(), line[597:627].strip(), line[627:629].strip(), line[629:632].strip(),
                             line[637:641].strip(), line[641:644].strip(), line[644:674].strip(), line[674:678].strip(),
                             line[678:681].strip(), line[681:711].strip(), line[711:712].strip(), line[712:716].strip(),
                             line[716:746].strip(), line[746:762].strip(), line[762:778].strip(), line[778:782].strip(),
                             line[782:812].strip(), line[812:828].strip(), line[828:832].strip(), line[832:833].strip(),
                             line[833:840].strip(), line[840:842].strip(), line[842:857].strip(), line[857:864].strip(),
                             line[864:877].strip(), line[877:878].strip(), line[878:879].strip(), line[879:880].strip(),
                             line[880:881].strip(), line[881:883].strip(), line[883:884].strip(), line[884:892].strip(),
                             line[892:900].strip(), line[900:940].strip(), line[940:945].strip(), line[945:950].strip(),
                             line[950:958].strip(), line[958:966].strip(), line[966:973].strip(), line[973:980].strip(),
                             line[980:981].strip(), line[981:988].strip(), line[988:995].strip(), line[995:999].strip(),
                             line[999:1002].strip(), line[1002:1003].strip(), line[1003:1004].strip(),
                             line[1061:1071].strip(), line[1071:1087].strip(), line[1087:1097].strip(),
                             line[1097:1113].strip(), line[1113:1123].strip(), line[1123:1124].strip(),
                             line[1124:1136].strip(), line[1136:1207].strip(), line[1207:1210].strip(),
                             line[1210:1217].strip(), line[1217:1529].strip()]) + '"'
                        nfdc_file.write(nfdc_line)
                line = aptfile.readline()
                """
                airport = dict()
                airport["RECORD TYPE INDICATOR"] = line[0:3].strip()
                airport["LANDING FACILITY SITE NUMBER"] = line[3:14].strip()
                airport["LANDING FACILITY TYPE"] = line[14:27].strip()
                airport["LOCATION IDENTIFIER"] = line[27:31].strip()
                airport["INFORMATION EFFECTIVE DATE (MM/DD/YYYY)"] = line[31:41].strip()
                airport["FAA REGION CODE"] = line[41:44].strip()
                airport["FAA DISTRICT OR FIELD OFFICE CODE"] = line[44:48].strip()
                airport["ASSOCIATED STATE POST OFFICE CODE"] = line[48:50].strip()
                airport["ASSOCIATED STATE NAME"] = line[50:70].strip()
                airport["ASSOCIATED COUNTY (OR PARISH) NAME"] = line[70:91].strip()
                airport["ASSOCIATED COUNTY'S STATE (POST OFFICE CODE)"] = line[91:93].strip()
                airport["ASSOCIATED CITY NAME"] = line[93:133].strip()
                airport["OFFICIAL FACILITY NAME"] = line[133:183].strip()
                airport["AIRPORT OWNERSHIP TYPE"] = line[183:185].strip()
                airport["FACILITY USE"] = line[185:187].strip()
                airport["FACILITY OWNER'S NAME"] = line[187:222].strip()
                airport["OWNER'S ADDRESS"] = line[222:294].strip()
                airport["OWNER'S CITY, STATE AND ZIP CODE"] = line[294:339].strip()
                airport["FACILITY MANAGER'S NAME"] = line[355:390].strip()
                airport["MANAGER'S ADDRESS"] = line[390:462].strip()
                airport["MANAGER'S CITY, STATE AND ZIP CODE"] = line[462:507].strip()
                airport["AIRPORT REFERENCE POINT LATITUDE (FORMATTED)"] = line[523:538].strip()
                airport["AIRPORT REFERENCE POINT LATITUDE (SECONDS)"] = line[538:550].strip()
                airport["AIRPORT REFERENCE POINT LONGITUDE (FORMATTED)"] = line[550:565].strip()
                airport["AIRPORT REFERENCE POINT LONGITUDE (SECONDS)"] = line[565:577].strip()
                airport["AIRPORT REFERENCE POINT DETERMINATION METHOD"] = line[577:578].strip()
                airport["AIRPORT ELEVATION DETERMINATION METHOD"] = line[585:586].strip()
                airport["MAGNETIC VARIATION AND DIRECTION"] = line[586:589].strip()
                airport["MAGNETIC VARIATION EPOCH YEAR"] = line[589:593].strip()
                airport["AERONAUTICAL SECTIONAL CHART ON WHICH FACILITY"] = line[597:627].strip()
                airport["DISTANCE FROM CENTRAL BUSINESS DISTRICT OF"] = line[627:629].strip()
                airport["DIRECTION OF AIRPORT FROM CENTRAL BUSINESS"] = line[629:632].strip()
                airport["BOUNDARY ARTCC IDENTIFIER"] = line[637:641].strip()
                airport["BOUNDARY ARTCC (FAA) COMPUTER IDENTIFIER"] = line[641:644].strip()
                airport["BOUNDARY ARTCC NAME"] = line[644:674].strip()
                airport["RESPONSIBLE ARTCC IDENTIFIER"] = line[674:678].strip()
                airport["RESPONSIBLE ARTCC (FAA) COMPUTER IDENTIFIER"] = line[678:681].strip()
                airport["RESPONSIBLE ARTCC NAME"] = line[681:711].strip()
                airport["TIE-IN FSS PHYSICALLY LOCATED ON FACILITY"] = line[711:712].strip()
                airport["TIE-IN FLIGHT SERVICE STATION (FSS) IDENTIFIER"] = line[712:716].strip()
                airport["TIE-IN FSS NAME"] = line[716:746].strip()
                airport["LOCAL PHONE NUMBER FROM AIRPORT TO FSS"] = line[746:762].strip()
                airport["TOLL FREE PHONE NUMBER FROM AIRPORT TO FSS"] = line[762:778].strip()
                airport["ALTERNATE FSS IDENTIFIER"] = line[778:782].strip()
                airport["ALTERNATE FSS NAME"] = line[782:812].strip()
                airport["TOLL FREE PHONE NUMBER FROM AIRPORT TO"] = line[812:828].strip()
                airport["IDENTIFIER OF THE FACILITY RESPONSIBLE FOR"] = line[828:832].strip()
                airport["AVAILABILITY OF NOTAM 'D' SERVICE AT AIRPORT"] = line[832:833].strip()
                airport["AIRPORT ACTIVATION DATE (MM/YYYY)"] = line[833:840].strip()
                airport["AIRPORT STATUS CODE"] = line[840:842].strip()
                airport["AIRPORT ARFF CERTIFICATION TYPE AND DATE"] = line[842:857].strip()
                airport["NPIAS/FEDERAL AGREEMENTS CODE"] = line[857:864].strip()
                airport["AIRPORT AIRSPACE ANALYSIS DETERMINATION"] = line[864:877].strip()
                airport["FACILITY HAS BEEN DESIGNATED BY THE U.S. TREASURY"] = line[877:878].strip()
                airport["FACILITY HAS BEEN DESIGNATED BY THE U.S. TREASURY"] = line[878:879].strip()
                airport["FACILITY HAS MILITARY/CIVIL JOINT USE AGREEMENT"] = line[879:880].strip()
                airport["AIRPORT HAS ENTERED INTO AN AGREEMENT THAT"] = line[880:881].strip()
                airport["AIRPORT INSPECTION METHOD"] = line[881:883].strip()
                airport["AGENCY/GROUP PERFORMING PHYSICAL INSPECTION"] = line[883:884].strip()
                airport["LAST PHYSICAL INSPECTION DATE (MMDDYYYY)"] = line[884:892].strip()
                airport["LAST DATE INFORMATION REQUEST WAS COMPLETED"] = line[892:900].strip()
                airport["FUEL TYPES AVAILABLE FOR PUBLIC USE AT THE"] = line[900:940].strip()
                airport["AIRFRAME REPAIR SERVICE AVAILABILITY/TYPE"] = line[940:945].strip()
                airport["POWER PLANT (ENGINE) REPAIR AVAILABILITY/TYPE"] = line[945:950].strip()
                airport["TYPE OF BOTTLED OXYGEN AVAILABLE (VALUE REPRESENTS"] = line[950:958].strip()
                airport["TYPE OF BULK OXYGEN AVAILABLE (VALUE REPRESENTS"] = line[958:966].strip()
                airport["AIRPORT LIGHTING SCHEDULE"] = line[966:973].strip()
                airport["BEACON LIGHTING SCHEDULE"] = line[973:980].strip()
                airport["AIR TRAFFIC CONTROL TOWER LOCATED ON AIRPORT"] = line[980:981].strip()
                airport["UNICOM FREQUENCY AVAILABLE AT THE AIRPORT"] = line[981:988].strip()
                airport["COMMON TRAFFIC ADVISORY FREQUENCY (CTAF)"] = line[988:995].strip()
                airport["SEGMENTED CIRCLE AIRPORT MARKER SYSTEM ON THE AIRPORT"] = line[995:999].strip()
                airport["LENS COLOR OF OPERABLE BEACON LOCATED ON THE AIRPORT"] = line[999:1002].strip()
                airport["LANDING FEE CHARGED TO NON-COMMERCIAL USERS OF"] = line[1002:1003].strip()
                airport['A "Y" IN THIS FIELD INDICATES THAT THE LANDING'] = line[1003:1004].strip()
                airport["12-MONTH ENDING DATE ON WHICH ANNUAL OPERATIONS DATA"] = line[1061:1071].strip()
                airport["AIRPORT POSITION SOURCE"] = line[1071:1087].strip()
                airport["AIRPORT POSITION SOURCE DATE (MM/DD/YYYY)"] = line[1087:1097].strip()
                airport["AIRPORT ELEVATION SOURCE"] = line[1097:1113].strip()
                airport["AIRPORT ELEVATION SOURCE DATE (MM/DD/YYYY)"] = line[1113:1123].strip()
                airport["CONTRACT FUEL AVAILABLE"] = line[1123:1124].strip()
                airport["TRANSIENT STORAGE FACILITIES"] = line[1124:1136].strip()
                airport["OTHER AIRPORT SERVICES AVAILABLE"] = line[1136:1207].strip()
                airport["WIND INDICATOR"] = line[1207:1210].strip()
                airport["ICAO IDENTIFIER"] = line[1210:1217].strip()
                airport["AIRPORT RECORD FILLER (BLANK)"] = line[1217:1529].strip()
                """

            except:
                """This is a dumb solution."""
                print('error at line {}, skipped:'.format(i))
                print(line)
                line = aptfile.readline()
                continue
    print(i)
    nfdc_file.close()

    df = pd.read_csv(nfdc_filename)

    us_airports = df[['LOCATION IDENTIFIER',
                       'OFFICIAL FACILITY NAME',
                       'ASSOCIATED CITY NAME',
                       'ASSOCIATED STATE POST OFFICE CODE',
                       'AIRPORT REFERENCE POINT LATITUDE (SECONDS)',
                       'AIRPORT REFERENCE POINT LATITUDE (FORMATTED)',
                       'AIRPORT REFERENCE POINT LONGITUDE (SECONDS)',
                       'AIRPORT REFERENCE POINT LONGITUDE (FORMATTED)',
                       'AIRPORT STATUS CODE',
                       'AIRPORT ACTIVATION DATE (MM/YYYY)',
                       'LAST PHYSICAL INSPECTION DATE (MMDDYYYY)',
                       'LANDING FACILITY TYPE',
                       'BOUNDARY ARTCC IDENTIFIER',
                       'RESPONSIBLE ARTCC IDENTIFIER',
                       'AIR TRAFFIC CONTROL TOWER LOCATED ON AIRPORT',
                       'LENS COLOR OF OPERABLE BEACON LOCATED ON THE AIRPORT',
                       'ICAO IDENTIFIER'
                       ]]

    return us_airports

if __name__ == '__main__':

    lat_dms, lon_dms = ll_decimal_to_dms(22.125, -22.125)
    assert(lat_dms == ['N', 22.0, 7.0, 30.0])
    assert(lon_dms == ['W', 22.0, 7.0, 30.0])

    usgs_airports = get_usgs_airport_list(r'usgs\usgs_tran_national_AirportPoint.csv')
    usgs_airport_df = airports_to_df(usgs_airports)
    bts_airports = get_bts_airport_list(r'bts\787626600_T_MASTER_CORD.zip')
    nfdc_airports = get_nfdc_airport_list(r'nfdc\APT.zip')
    bts_airport_df = airports_to_df(bts_airports)
    nfdc_airport_df = airports_to_df(nfdc_airports)
    # for lexically similar airports, what's the physical distance?
    # get suspicious pairs (similar airport name or id, relatively large distance (try: 10km, 50km, 100km)
    # Prep: convert dms to string
    # Check for:
    # Transposed elements (minutes swapped with seconds)
    # Transposed digits (first and second digit swapped within an element)
    print('Checking matches...')
    recorded_airport_count = 0
    for airport in bts_airports:
        state_airports = nfdc_airport_df[(nfdc_airport_df['state'] == airport.state)]
        id_matches = state_airports[(state_airports['name'] == airport.name)]
        match_count = len(id_matches)
        if match_count == 1:
            if ((airport.lat_dms_string[0:-1] != id_matches.lat_dms_string.item()[0:-1]) or
                (airport.lon_dms_string[0:-1] != id_matches.lon_dms_string.item()[0:-1])):
                distance = haversine_np(airport.lon, airport.lat, id_matches.lon.item(), id_matches.lat.item())
                if distance > 5: # 0.1:
                    recorded_airport_count = recorded_airport_count + 1
                    print('Found match for {}: {}'.format(airport.name,id_matches.name.item()))
                    print('\tlat\n\t{}:nfdc\n\t{}:bts'.format(airport.lat_dms_string, id_matches.lat_dms_string.item()))
                    print('\tlon\n\t{}:nfdc\n\t{}:bts'.format(airport.lon_dms_string, id_matches.lon_dms_string.item()))
                    print('\t{:2.2f} km apart'.format(distance))
                    # a = sorted(airport.lat_dms_string[1:])
                    print('\tnfdc: {}\n\tbts :{}'.format(airport.city, id_matches.city.item()))
        elif match_count < 1:
            continue
            print('Match not found.')
        elif match_count > 1:
            print('Too many matches for {}.'.format(airport.name))
            id_matches = state_airports[(state_airports['name'] == airport.name) &
                                        (state_airports['city'] == airport.city)]
            match_count = len(id_matches)
            if match_count == 1:
                recorded_airport_count = recorded_airport_count + 1
                print('Found city match for {}: {}'.format(airport.name,id_matches.name.item()))
                print('\tlat\n\t{}:nfdc\n\t{}:bts'.format(airport.lat_dms_string, id_matches.lat_dms_string.item()))
                print('\tlon\n\t{}:nfdc\n\t{}:bts'.format(airport.lon_dms_string, id_matches.lon_dms_string.item()))
                print('\t{:2.2f} km apart'.format(distance))
                # a = sorted(airport.lat_dms_string[1:])
                print('\tnfdc: {}\n\tbts :{}'.format(airport.city, id_matches.city.item()))


    len(bts_airports)
    print('{} matches found.'.format(recorded_airport_count))

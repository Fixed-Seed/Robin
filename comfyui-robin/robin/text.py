import re

ONES = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen '
        'nineteen').split()
TENS = ('', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety')
SCALES = ('', ' thousand', ' million', ' billion', ' trillion')
IRREGULAR = dict(one='first', two='second', three='third', five='fifth', eight='eighth', nine='ninth', twelve='twelfth')
MONTHS = ('January February March April May June July August September October November December').split()
MONTH = re.compile(r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|'
                   r'Nov(?:ember)?|Dec(?:ember)?)')


def number(n):
    if n < 0: return 'minus ' + number(-n)
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + ('-' + ONES[n % 10] if n % 10 else '')
    if n < 1000: return ONES[n // 100] + ' hundred' + (' ' + number(n % 100) if n % 100 else '')
    if n >= 10 ** 15: return digits(str(n))
    out, i = [], 0
    while n:
        n, r = divmod(n, 1000)
        if r: out.append(number(r) + SCALES[i])
        i += 1
    return ' '.join(reversed(out))


def digits(s, zero='zero'):
    return ' '.join(zero if c == '0' else ONES[int(c)] for c in s if c.isdigit())


def ordinal(n):
    head, last = re.match(r'(.*?)([a-z]+)$', number(n)).groups()
    return head + (IRREGULAR.get(last) or (last[:-1] + 'ieth' if last.endswith('y') else last + 'th'))


def year(n):
    if not 1000 <= n <= 9999 or n % 1000 == 0 or 2000 < n < 2010: return number(n)
    hi, lo = divmod(n, 100)
    return number(hi) + (' hundred' if lo == 0 else ' oh ' + ONES[lo] if lo < 10 else ' ' + number(lo))


def pairs(s):
    if len(s) > 1 and s[0] == '0' or len(s) > 4: return digits(s)
    n = int(s)
    if len(s) == 4: return year(n)
    if len(s) < 3 or n % 100 == 0: return number(n)
    return ONES[n // 100] + (' oh ' + ONES[n % 100] if n % 100 < 10 else ' ' + number(n % 100))


def cardinal(s):
    s = s.replace(',', '')
    return digits(s) if len(s) > 1 and s[0] == '0' or len(s) > 15 else number(int(s))


def decimal(s):
    whole, _, frac = s.replace(',', '').partition('.')
    words = cardinal(whole) if whole else ''
    return (words + ' point ' if words else 'point ') + digits(frac) if frac else words


def plural(words):
    return words[:-1] + 'ies' if words.endswith('y') else words + ('es' if words.endswith(('x', 's')) else 's')


def fraction(n, d, of_unit=None):
    name = {2: 'half', 4: 'quarter'}.get(d) or ordinal(d)
    if of_unit:
        if n == 1: return f'half a {of_unit}' if d == 2 else ('an ' if name[0] in 'aeiou' else 'a ') + f'{name} {of_unit}'
        return f'{number(n)} {"halves" if d == 2 else name + "s"} of a {of_unit}'
    return f'{number(n)} ' + (name if n == 1 else 'halves' if d == 2 else name + 's')


UNITS = {
    'mm': 'millimeter', 'cm': 'centimeter', 'km': 'kilometer', 'm': 'meter', 'nm': 'nanometer', 'µm': 'micrometer',
    'mi': 'mile', 'ft': ('foot', 'feet'), 'yd': 'yard', 'in.': ('inch', 'inches'), 'ly': 'light-year',
    'mg': 'milligram', 'mcg': 'microgram', 'µg': 'microgram', 'g': 'gram', 'kg': 'kilogram', 'lb': 'pound', 'lbs': 'pound',
    'oz': 'ounce', 'fl oz': 'fluid ounce', 'ml': 'milliliter', 'mL': 'milliliter', 'L': 'liter', 'dL': 'deciliter', 'dl': 'deciliter',
    'tsp': 'teaspoon', 'tbsp': 'tablespoon', 'Tbsp': 'tablespoon', 'TBSP': 'tablespoon', 'qt': 'quart', 'gal': 'gallon',
    'ms': 'millisecond', 'sec': 'second', 'secs': 'second', 'min': 'minute', 'mins': 'minute', 'hr': 'hour', 'hrs': 'hour',
    'yr': 'year', 'yrs': 'year', 'wk': 'week', 'wks': 'week',
    'Hz': ('hertz', 'hertz'), 'kHz': ('kilohertz', 'kilohertz'), 'MHz': ('megahertz', 'megahertz'), 'GHz': ('gigahertz', 'gigahertz'),
    'KB': 'kilobyte', 'kB': 'kilobyte', 'MB': 'megabyte', 'GB': 'gigabyte', 'TB': 'terabyte', 'PB': 'petabyte',
    'Kb': 'kilobit', 'kb': 'kilobit', 'Mb': 'megabit', 'Gb': 'gigabit',
    'kbps': ('kilobit per second', 'kilobits per second'), 'Kbps': ('kilobit per second', 'kilobits per second'),
    'Mbps': ('megabit per second', 'megabits per second'), 'Gbps': ('gigabit per second', 'gigabits per second'),
    'MB/s': ('megabyte per second', 'megabytes per second'), 'GB/s': ('gigabyte per second', 'gigabytes per second'),
    'W': 'watt', 'kW': 'kilowatt', 'MW': 'megawatt', 'GW': 'gigawatt', 'Wh': 'watt-hour', 'kWh': 'kilowatt-hour',
    'MWh': 'megawatt-hour', 'GWh': 'gigawatt-hour', 'mAh': 'milliamp-hour', 'V': 'volt', 'mV': 'millivolt', 'kV': 'kilovolt',
    'mA': 'milliamp', 'kcal': 'kilocalorie', 'cal': 'calorie', 'kJ': 'kilojoule',
    'Pa': 'pascal', 'kPa': 'kilopascal', 'hPa': 'hectopascal', 'psi': ('P S I', 'P S I'), 'mmHg': ('millimeter of mercury',
                                                                                                 'millimeters of mercury'),
    'mph': ('mile per hour', 'miles per hour'), 'kph': ('kilometer per hour', 'kilometers per hour'),
    'km/h': ('kilometer per hour', 'kilometers per hour'), 'm/s': ('meter per second', 'meters per second'),
    'mg/dL': ('milligram per deciliter', 'milligrams per deciliter'), 'mmol/L': ('millimole per liter', 'millimoles per liter'),
    'rpm': ('R P M', 'R P M'), 'RPM': ('R P M', 'R P M'), 'bpm': ('B P M', 'B P M'), 'BPM': ('B P M', 'B P M'),
    'fps': ('F P S', 'F P S'), 'FPS': ('F P S', 'F P S'), 'dB': 'decibel', 'dpi': ('D P I', 'D P I'), 'DPI': ('D P I', 'D P I'),
    'ppi': ('P P I', 'P P I'), 'px': 'pixel', 'IU': ('I U', 'I U'), 'N·m': ('newton-meter', 'newton-meters'), 'Nm': ('newton-meter', 'newton-meters'),
    'lb-ft': ('pound-foot', 'pound-feet'), 'lb·ft': ('pound-foot', 'pound-feet'),
}
POWERS = {'²': 'square', '³': 'cubic', '^2': 'square', '^3': 'cubic'}
CURRENCIES = {
    '$': ('dollar', 'dollars', 'cent', 'cents'), 'US$': ('dollar', 'dollars', 'cent', 'cents'),
    'A$': ('Australian dollar', 'Australian dollars', 'cent', 'cents'), 'AU$': ('Australian dollar', 'Australian dollars', 'cent', 'cents'),
    'C$': ('Canadian dollar', 'Canadian dollars', 'cent', 'cents'), 'CA$': ('Canadian dollar', 'Canadian dollars', 'cent', 'cents'),
    'NZ$': ('New Zealand dollar', 'New Zealand dollars', 'cent', 'cents'), 'HK$': ('Hong Kong dollar', 'Hong Kong dollars', 'cent', 'cents'),
    'S$': ('Singapore dollar', 'Singapore dollars', 'cent', 'cents'), 'R$': ('real', 'reais', 'centavo', 'centavos'),
    'MX$': ('peso', 'pesos', 'centavo', 'centavos'), '€': ('euro', 'euros', 'cent', 'cents'), '£': ('pound', 'pounds', 'penny', 'pence'),
    '¥': ('yen', 'yen', None, None), '₹': ('rupee', 'rupees', 'paisa', 'paise'), '₩': ('won', 'won', None, None),
    '₽': ('ruble', 'rubles', 'kopek', 'kopeks'), '₱': ('peso', 'pesos', 'centavo', 'centavos'), '₺': ('lira', 'lira', 'kurus', 'kurus'),
}
CODES = {'USD': ('U S dollar', 'U S dollars'), 'EUR': ('euro', 'euros'), 'GBP': ('pound', 'pounds'), 'JPY': ('yen', 'yen'),
         'CHF': ('Swiss franc', 'Swiss francs'), 'CAD': ('Canadian dollar', 'Canadian dollars'), 'AUD': ('Australian dollar',
         'Australian dollars'), 'NZD': ('New Zealand dollar', 'New Zealand dollars'), 'INR': ('rupee', 'rupees'), 'CNY': ('yuan', 'yuan'),
         'MXN': ('peso', 'pesos'), 'BRL': ('real', 'reais'), 'KRW': ('won', 'won'), 'SEK': ('krona', 'kronor'), 'NOK': ('krone', 'kroner'),
         'DKK': ('krone', 'kroner'), 'BTC': ('bitcoin', 'bitcoin'), 'ETH': ('ether', 'ether'), 'HKD': ('Hong Kong dollar', 'Hong Kong dollars'),
         'SGD': ('Singapore dollar', 'Singapore dollars'), 'ZAR': ('rand', 'rand'), 'TRY': ('lira', 'lira'), 'PLN': ('zloty', 'zlotys')}
SCALE_WORDS = {'k': 'thousand', 'K': 'thousand', 'm': 'million', 'M': 'million', 'mm': 'million', 'mn': 'million', 'MM': 'million',
               'b': 'billion', 'B': 'billion', 'bn': 'billion', 'Bn': 'billion', 't': 'trillion', 'T': 'trillion', 'tn': 'trillion'}
AS_WORDS = {
    'NASA': 'NASA', 'NATO': 'NATO', 'UNESCO': 'UNESCO', 'UNICEF': 'UNICEF', 'FIFA': 'FIFA', 'FEMA': 'FEMA', 'OPEC': 'OPEC',
    'NAFTA': 'NAFTA', 'POTUS': 'POTUS', 'SCOTUS': 'SCOTUS', 'SWAT': 'SWAT', 'AIDS': 'AIDS', 'COVID': 'COVID', 'SARS': 'SARS',
    'SCUBA': 'scuba', 'LASER': 'laser', 'RADAR': 'radar', 'SONAR': 'sonar', 'CAPTCHA': 'captcha', 'YOLO': 'yolo', 'FOMO': 'fomo',
    'ROM': 'rom', 'RAM': 'ram', 'PIN': 'pin', 'ZIP': 'zip', 'SIM': 'sim', 'GIF': 'gif', 'JPEG': 'jay-peg', 'JSON': 'jay-son',
    'GUI': 'gooey', 'ASCII': 'ask-ee', 'IKEA': 'Ikea', 'LEGO': 'Lego', 'OK': 'OK', 'TBD': 'T B D', 'WWII': 'World War Two',
    'WWI': 'World War One', 'ASAP': 'A S A P', 'NIMBY': 'nimby', 'IMAX': 'IMAX', 'AWOL': 'AWOL', 'MASH': 'MASH', 'GOAT': 'GOAT',
    'UCLA': 'U C L A', 'ACLU': 'A C L U', 'AARP': 'A A R P', 'IEEE': 'I triple E', 'NAACP': 'N double A C P', 'USAF': 'U S A F',
    'OLED': 'O L E D', 'SQL': 'S Q L', 'AMOLED': 'A M O L E D', 'URL': 'U R L', 'UI': 'U I', 'API': 'A P I',
    'DOS': 'doss', 'LAN': 'lan', 'VLAN': 'vee-lan', 'NAS': 'nass', 'SCSI': 'scuzzy', 'NGINX': 'engine-ex', 'WYSIWYG': 'wizzy-wig',
    'CMOS': 'see-moss', 'BIOS': 'bye-oss', 'SATA': 'say-tuh', 'RAID': 'raid', 'VRAM': 'vee-ram', 'DRAM': 'dee-ram', 'SRAM': 'ess-ram',
    'SDRAM': 'ess-dee-ram', 'IMAP': 'eye-map', 'YAML': 'yamel', 'POP': 'pop', 'WAV': 'wave', 'OGG': 'ogg', 'FLAC': 'flack',
    'MPEG': 'em-peg', 'MIDI': 'middy', 'LUFS': 'luffs', 'LUT': 'lut', 'SMPTE': 'simp-tee', 'PAL': 'pal', 'HUD': 'hud', 'LARP': 'larp',
    'MOBA': 'moba', 'LIDAR': 'lidar', 'PAC': 'pack', 'TED': 'Ted', 'HIPAA': 'hippa',
}
ONSETS = set('BL BR CH CL CR DR DW FL FR GL GN GR KL KN KR KW PH PL PR PS QU SC SH SK SL SM SN SP ST SW TH TR TW WH WR SCH SCR SHR SPL '
             'SPR STR THR'.split())
CODAS = set('CH CK CT FT GH LB LD LF LK LL LM LP LT MB MP ND NG NK NS NT PH PT RB RC RD RF RG RK RL RM RN RP RS RT SH SK SP SS ST TH TS '
            'TZ XT'.split())


def pronounceable(w):
    if not re.search('[AEIOUY]', w): return False
    if len(w) >= 6: return True
    w = re.sub(r'(?<=[AEIOU])[HWY]', 'E', w)
    for run in re.findall(r'[^AEIOUY]+', w):
        if len(run) > 3: return False
    m = re.match(r'[^AEIOUY]+', w)
    if m and len(m.group(0)) > 1 and m.group(0) not in ONSETS: return False
    m = re.search(r'[^AEIOUY]+$', w)
    if m and len(m.group(0)) > 1 and m.group(0) not in CODAS: return False
    return all(len(r) < 2 or r in ONSETS or r in CODAS or r[:-1] in CODAS and r[-1:] + 'R' in ONSETS | {r[-1:] + 'R'}
               for r in re.findall(r'(?<=[AEIOUY])[^AEIOUY]{2,}(?=[AEIOUY])', w)) or len(w) >= 6
RESPELLING_CUES = re.compile(r'\b(?:pronounced?|pronouncing|pronunciation|rhymes?|stress(?:ed)?|syllables?|spelled|accent on)\b', re.I)
RESPELLING = re.compile(r'\b[A-Za-z]+(?:-[A-Za-z]+)+\b')
RESPELL = {'codec': 'koh-deck', 'codecs': 'koh-decks', 'bitrate': 'bit rate', 'bitrates': 'bit rates', 'iOS': 'eye O S',
           'MySQL': 'my S Q L', 'NoSQL': 'no S Q L', 'PhD': 'P H D', 'PhDs': 'P H Ds',
           'macOS': 'mac O S', 'iPadOS': 'eye-pad O S', 'tvOS': 'T V O S', 'watchOS': 'watch O S'}
COMMON = set('''a about after again all also always an and any are as at away back bad be because been before being best better big
both but by call came can come could day did do does done down each even ever every far few find first for from get give go goes going
gone good got great had has have he her here him his how i if in into is its just keep kind know last least left less let like
little long look lot love made make man many may me might mine more most much must my need never new next nice no none nor not
nothing now of off oh old on once one only or our out over own please put real really right run said same saw say see seen she
should show since so some still stop such sure take tell than thank thanks that the their them then there these they thing think
this those though through time to too took true try turn two under until up upon use very wait want was watch way we well went
were what when where which while who whole why will wish with without wow yeah yes yet you your yours hey help hello go stay run
listen look out right left never ever again done over here there love hate fire free gold big huge boom bang ugh omg wait'''.split())
REGNAL = set('''Henry Louis Elizabeth George Charles Edward William James Richard John Mary Victoria Anne Philip Felipe Juan Carlos
Frederick Friedrich Wilhelm Napoleon Peter Catherine Ivan Nicholas Alexander Alfonso Ferdinand Leopold Rudolf Francis Franz Pius
Benedict Gregory Leo Innocent Clement Boniface Urban Paul Ramesses Ramses Thutmose Amenhotep Ptolemy Constantine Justinian Otto
Gustav Gustavus Christian Olaf Haakon Harald Pedro Manuel Afonso Henri Ludwig Maximilian Selim Suleiman Mehmed Murad Ahmed Rama
David Robert Kamehameha Willem Carl Gustaf Harold Edmund Stephen Matilda Sancho Wenceslaus Sigismund Casimir Boleslaw Vladimir
Basil Michael Andronikos Theodore Sixtus Julius Martin Nicolas Augustus Antiochus Seleucus Darius Xerxes Cyrus Artaxerxes'''.split())
ROMAN_AFTER = set('''War Part Chapter Act Book Volume Vol Phase Episode Season Level Stage Type Class Gen Generation Mark Bowl Rocky
Fantasy Edition Round Wave Tier Grade Section Article Title Division Appendix Psalm Canto Scene Movement Symphony Op Opus Unit Year
Form Saga Quest Wars Kombat Rising Revelations Legacy Origins Collection Chronicles Trilogy Remastered Zero Force Command'''.split())
STATES = set('AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD '
             'TN TX UT VT VA WA WV WI WY DC'.split())
TLD_WORDS = set('com net org edu gov mil int info biz shop store blog news art design games site online tech app dev co me so to page '
                'live media cloud space world club life today zone'.split())
TLDS = TLD_WORDS | set('io ai uk us tv gg ly fm ca de jp fr au nz in xyz eu es it nl se no ch be at dk fi pl ru br mx cn kr'.split())
EXTENSIONS = {'jpg': 'J P G', 'jpeg': 'jay-peg', 'png': 'P N G', 'gif': 'gif', 'webp': 'web P', 'heic': 'H E I C', 'svg': 'S V G',
              'pdf': 'P D F', 'doc': 'doc', 'docx': 'doc X', 'xls': 'X L S', 'xlsx': 'X L S X', 'ppt': 'P P T', 'pptx': 'P P T X',
              'txt': 'T X T', 'csv': 'C S V', 'md': 'M D', 'json': 'jay-son', 'xml': 'X M L', 'yaml': 'yaml', 'yml': 'Y M L',
              'html': 'H T M L', 'htm': 'H T M', 'css': 'C S S', 'js': 'J S', 'ts': 'T S', 'py': 'P Y', 'rb': 'R B', 'go': 'go',
              'rs': 'R S', 'java': 'java', 'cpp': 'C P P', 'sh': 'S H', 'exe': 'E X E', 'dmg': 'D M G', 'apk': 'A P K',
              'zip': 'zip', 'rar': 'rar', 'tar': 'tar', 'gz': 'G Z', 'iso': 'I S O', 'mp3': 'M P three', 'mp4': 'M P four',
              'm4a': 'M four A', 'wav': 'wave', 'flac': 'flack', 'ogg': 'ogg', 'mov': 'mov', 'avi': 'A V I', 'mkv': 'M K V',
              'webm': 'web M', 'srt': 'S R T', 'epub': 'E pub', 'ttf': 'T T F', 'otf': 'O T F', 'psd': 'P S D', 'raw': 'raw',
              'arw': 'A R W', 'cr2': 'C R two', 'cr3': 'C R three', 'nef': 'N E F', 'dng': 'D N G', 'tif': 'tiff', 'tiff': 'tiff',
              'bmp': 'B M P', 'log': 'log', 'ini': 'I N I', 'cfg': 'C F G', 'sql': 'S Q L', 'db': 'D B', 'bak': 'back',
              'pages': 'pages', 'key': 'key', 'numbers': 'numbers', 'ipynb': 'I P Y N B', 'kt': 'K T', 'swift': 'swift',
              'odt': 'O D T', 'ods': 'O D S', 'odp': 'O D P', 'rtf': 'R T F'}
ABBREVIATIONS = {
    'Mr.': 'Mister', 'Mrs.': 'Missus', 'Ms.': 'Miz', 'Prof.': 'Professor', 'Lt.': 'Lieutenant', 'Sgt.': 'Sergeant',
    'Capt.': 'Captain', 'Col.': 'Colonel', 'Gen.': 'General', 'Gov.': 'Governor', 'Sen.': 'Senator', 'Rep.': 'Representative',
    'Pres.': 'President', 'Rev.': 'Reverend', 'Hon.': 'Honorable', 'Supt.': 'Superintendent', 'Cmdr.': 'Commander',
    'vs.': 'versus', 'vs': 'versus', 'etc.': 'et cetera', 'e.g.': 'for example', 'i.e.': 'that is', 'approx.': 'approximately',
    'dept.': 'department', 'Dept.': 'Department', 'est.': 'established', 'Est.': 'Established', 'misc.': 'miscellaneous',
    'govt.': 'government', 'Govt.': 'Government', 'Inc.': 'Incorporated', 'Ltd.': 'Limited', 'Corp.': 'Corporation',
    'Bros.': 'Brothers', 'Assn.': 'Association', 'Univ.': 'University', 'Intl.': 'International', 'Natl.': 'National',
    'Ave.': 'Avenue', 'Blvd.': 'Boulevard', 'Rd.': 'Road', 'Ln.': 'Lane', 'Hwy.': 'Highway', 'Rte.': 'Route', 'Pkwy.': 'Parkway',
    'Apt.': 'Apartment', 'Ste.': 'Suite', 'Mt.': 'Mount', 'Ft.': 'Fort', 'Jr.': 'Junior', 'Sr.': 'Senior', 'Ph.D.': 'P H D',
    'Fig.': 'Figure', 'fig.': 'figure', 'Vol.': 'Volume', 'vol.': 'volume', 'Ch.': 'Chapter', 'ch.': 'chapter', 'pp.': 'pages',
    'pg.': 'page', 'max.': 'maximum', 'min.': 'minimum', 'avg.': 'average', 'temp.': 'temperature', 'ext.': 'extension',
    'Mon.': 'Monday', 'Tue.': 'Tuesday', 'Tues.': 'Tuesday', 'Wed.': 'Wednesday', 'Thu.': 'Thursday', 'Thur.': 'Thursday',
    'Thurs.': 'Thursday', 'Fri.': 'Friday', 'Sat.': 'Saturday', 'Sun.': 'Sunday',
    'w/': 'with', 'w/o': 'without', 'b/c': 'because', 'c/o': 'care of', 'N/A': 'not applicable', 'n/a': 'not applicable',
    'a/c': 'A C', 'A/C': 'A C', 'P/E': 'P E', 'TL;DR': 'T L D R', 'tl;dr': 'T L D R',
}
TITLES = {'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Lt.', 'Sgt.', 'Capt.', 'Col.', 'Gen.', 'Gov.', 'Sen.', 'Rep.', 'Pres.', 'Rev.', 'Hon.', 'Supt.',
          'Cmdr.', 'Mt.', 'Ft.'}
ENDS_SENTENCE = {'etc.', 'Inc.', 'Ltd.', 'Corp.', 'Bros.', 'Jr.', 'Sr.', 'Ave.', 'Blvd.', 'Rd.', 'Ln.', 'Hwy.', 'Pkwy.', 'Ph.D.',
                 'approx.', 'misc.', 'Assn.', 'Univ.'}
NETSPEAK = {'rn': 'right now', 'gtg': 'got to go', 'brb': 'be right back', 'idk': "I don't know", 'tbh': 'to be honest',
            'imo': 'in my opinion', 'imho': 'in my humble opinion', 'btw': 'by the way', 'irl': 'in real life', 'np': 'no problem',
            'thx': 'thanks', 'ty': 'thank you', 'pls': 'please', 'plz': 'please', 'ngl': 'not gonna lie', 'jk': 'just kidding',
            'nvm': 'never mind', 'ikr': 'I know, right', 'afaik': 'as far as I know', 'smh': 'S M H', 'lol': 'L O L', 'lmao': 'L M A O',
            'omg': 'O M G', 'wtf': 'W T F', 'fyi': 'F Y I', 'dm': 'D M', 'dms': 'D Ms', 'gg': 'G G', 'afk': 'A F K', 'ftw': 'for the win',
            'ngmi': 'N G M I', 'wagmi': 'wag me', 'bc': 'because', 'atm': 'at the moment', 'iirc': 'if I recall correctly',
            'ootd': 'outfit of the day', 'ianal': 'I A N A L', 'npc': 'N P C', 'npcs': 'N P Cs', 'eli5': 'E L I five', 'iykyk': 'I Y K Y K',
            'qotd': 'Q O T D', 'tfw': 'T F W', 'mfw': 'M F W', 'pov': 'P O V', 'pfp': 'P F P', 'ofc': 'of course', 'nsfw': 'N S F W',
            'omw': 'on my way', 'hmu': 'hit me up', 'wyd': 'what are you doing', 'hbu': 'how about you', 'ily': 'I love you',
            'idc': "I don't care", 'tmi': 'T M I', 'lmk': 'let me know', 'ttyl': 'talk to you later', 'nbd': 'no big deal',
            'wfh': 'work from home', 'lfg': 'L F G', 'icymi': 'in case you missed it', 'tbt': 'T B T', 'fwiw': "for what it's worth",
            'tbf': 'to be fair', 'tysm': 'thank you so much', 'pvp': 'P V P', 'pve': 'P V E', 'mmo': 'M M O', 'rpg': 'R P G',
            'fr': 'for real', 'ymmv': 'Y M M V', 'fyp': 'F Y P', 'istg': 'I swear to God', 'tl;dr': 'T L D R'}
KEYS = {'Cmd': 'Command', 'Ctrl': 'Control', 'Alt': 'Alt', 'Opt': 'Option', 'Option': 'Option', 'Shift': 'Shift', 'Win': 'Windows',
        'Fn': 'Function', 'Esc': 'Escape', 'Del': 'Delete', 'Tab': 'Tab', 'Enter': 'Enter', 'Return': 'Return', 'Space': 'Space',
        'Super': 'Super', 'Meta': 'Meta', 'Backspace': 'Backspace', 'PgUp': 'Page Up', 'PgDn': 'Page Down', 'Home': 'Home', 'End': 'End'}
YEAR_CUES = set('in since from by until till til circa c. ca. during before after around of year years spring summer fall autumn winter '
                'early late mid the and to through between class vintage born died founded built released'.split())
FRACTION_DENOMS = {2, 3, 4, 5, 6, 8, 10, 12, 16, 32, 64, 100}
CODE_WORDS = set('zip code pin id ext extension account acct order confirmation ticket serial reference ref tracking postcode passcode'.split())
STREET = r'(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Ct|Court|Pl|Place|Pkwy|Parkway|Hwy|Highway|Ter|Terrace|Cir|Circle|Trail)\b'
QUANTITY_NOUNS = set('people men women children kids staff crew personnel feet foot money cash fish sheep deer'.split())
NOT_PLURAL = set('''was is has does gives goes makes takes sees says gets comes means seems marks changes brings starts ends begins
sits stands looks feels works runs plays shows its this his hers as us yes plus minus versus whereas always perhaps unless besides
thus across less series news '''.split())
NUM = r'\d{1,3}(?:,\d{3})+|\d+'
DEC = rf'(?:{NUM})(?:\.\d+)?|\.\d+'
BEFORE = r'(?<![\w.,/$€£¥₹#@-])'


def _unit(u):
    v = UNITS[u]
    return v if isinstance(v, tuple) else (v, v + 's')


def letters(s):
    return ' '.join(s.upper())


def ident(s):
    out = []
    for part in re.split(r'([_.\-])', s):
        if part in ('_', '.', '-'): out.append({'_': 'underscore', '.': 'dot', '-': 'dash'}[part]); continue
        for run in re.findall(r'[A-Za-z]+|\d+', part):
            if run.isdigit(): out.append(year(int(run)) if len(run) == 4 and 1900 <= int(run) <= 2099 else
                                         cardinal(run) if len(run) <= 2 else digits(run))
            elif run.isupper() and len(run) > 1 and run not in AS_WORDS: out.append(letters(run))
            else: out.append(AS_WORDS.get(run, run))
    return ' '.join(out)


def alnum(s):
    out = []
    chem = re.fullmatch(r'(?:[A-Z][a-z]?\d*){2,}', s) and re.search(r'\d', s) and not re.search(r'[a-z]{2}', s)
    for part in s.split('-'):
        for run in re.findall(r'[A-Za-z]+|\d+\.\d+|\d+', part):
            if run[0].isdigit(): out.append(decimal(run) if '.' in run else pairs(run))
            elif chem or len(run) == 1 or run.isupper() and run not in AS_WORDS or len(run) <= 3 and not re.search(r'[aeiouyAEIOUY]', run):
                out.append(letters(run))
            else: out.append(AS_WORDS.get(run, run))
    return ' '.join(out)


def roman(s):
    vals = dict(I=1, V=5, X=10, L=50, C=100)
    total = 0
    for a, b in zip(s, s[1:] + ' '):
        total += -vals[a] if vals.get(b, 0) > vals[a] else vals[a]
    return total if 0 < total < 400 and re.fullmatch(r'C{0,3}(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})', s) else None


def _sub(pattern, fn, s, flags=0):
    return re.sub(pattern, fn, s, flags=flags)


def _money(m):
    neg, cur, a, sa, b, sb = m.group('neg'), m.group('cur'), m.group('a'), m.group('sa'), m.group('b'), m.group('sb')
    one, many, sub1, subs = CURRENCIES[cur]

    def amount(x, scale):
        if scale:
            return f'{decimal(x)} {SCALE_WORDS.get(scale, scale.lower())}', True
        whole, _, frac = x.replace(',', '').partition('.')
        if not sub1 or not frac: return decimal(x), (x.replace(',', '') != '1')
        cents = int((frac + '0')[:2])
        w = int(whole or 0)
        if not w: return f'{number(cents)} {sub1 if cents == 1 else subs}', None
        return f'{number(w)} {one if w == 1 else many}' + (f' and {number(cents)} {sub1 if cents == 1 else subs}' if cents else ''), None

    first, plural_a = amount(a, sa)
    out = ('minus ' if neg else '')
    if b:
        second, plural_b = amount(b, sb or sa)
        if plural_a is not None and plural_b is not None: return out + f'{first} to {second} {many}'
        if plural_a is not None: first += f' {many if plural_a else one}'
        if plural_b is not None: second += f' {many if plural_b else one}'
        return out + f'{first} to {second}'
    return out + (first if plural_a is None else f'{first} {many if plural_a else one}')


def _time(m):
    h, mm, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    hw = number(h % 12 or 12) if ap else number(h)
    if mm == 0: words = hw if ap else (f'{hw} hundred' if h >= 13 or h == 0 else f"{hw} o'clock")
    else: words = f'{hw} oh {ONES[mm]}' if mm < 10 else f'{hw} {number(mm)}'
    return words + (f' {ap[0].upper()} M' if ap else '')


def _duration(m):
    h, mi, se = int(m.group(1)), int(m.group(2)), int(m.group(3))
    parts = [f'{number(v)} {u}{"" if v == 1 else "s"}' for v, u in ((h, 'hour'), (mi, 'minute'), (se, 'second')) if v]
    return ', '.join(parts[:-1]) + (' and ' if len(parts) > 1 else '') + parts[-1] if parts else 'zero seconds'


def _date_words(mo, d, y=None, order='us'):
    month = MONTHS[mo - 1]
    out = f'{month} {ordinal(d)}' if order == 'us' else f'the {ordinal(d)} of {month}'
    if y is not None: out += ', ' + (year(y) if len(str(y)) == 4 else short_year(str(y).zfill(2)))
    return out


def short_year(yy):
    return 'oh ' + ONES[int(yy[1])] if yy[0] == '0' else number(int(yy))


def _month_index(name):
    return [m[:3] for m in MONTHS].index(name[:3].title()) + 1


def _unit_amount(m):
    a, b, u, pw = m.group('a'), m.group('b'), m.group('u'), m.group('pw')
    one, many = _unit(u)
    if pw:
        p = POWERS[pw]; one, many = f'{p} {one}', f'{p} {many}'
    first = decimal(a)
    if b: return f'{first} to {decimal(b)} {many}'
    return f'{first} {one if a == "1" else many}'


def _number_in_context(m):
    raw, s = m.group(0), m.string
    if len(raw) > 1 and raw[0] == '0': return digits(raw)
    before, after = s[:m.start()], s[m.end():]
    pm = re.search(r"([A-Za-z][\w.']*)\s+$", before)
    prev = pm.group(1) if pm else ''
    starts = not pm or re.search(r'(^|[.!?:;]\s+|["(]\s*)$', before[:pm.start()]) is not None
    nm = re.match(r'\s+([a-z]+)', after); nxt = nm.group(1) if nm else ''
    v = int(raw)
    counted = nxt in QUANTITY_NOUNS or nxt.endswith('s') and nxt not in NOT_PLURAL
    if prev.lower().rstrip('.') in CODE_WORDS and len(raw) >= 4 and not counted: return digits(raw)
    if 3 <= len(raw) <= 5 and re.match(r'\s+(?:[A-Z][\w]*\.?\s+){1,3}' + STREET, after): return pairs(raw)
    if len(raw) == 4:
        if prev.lower() in YEAR_CUES or not counted: return year(v)
        return number(v // 100) + ' hundred' if v % 100 == 0 and v % 1000 else number(v)
    if len(raw) == 3 and prev and (prev.isupper() and len(prev) > 1 or prev[0].isupper() and not starts):
        return pairs(raw)
    return number(v)


def _words(s):
    s = re.sub(r'(^|[.!?]["\')\]]?\s+)(?=[-$€£¥₹#~+]?\d)', _opener, s)
    s = _rules(s)
    return re.sub('\x01(.)', lambda m: m.group(1).upper(), s).replace('\x01', '')


SHORT_ABBR = re.compile(r'\b(?:[A-Z][a-z]{0,3}|no|vol|pp|pg|ch|fig|sec|art|ca|vs|ex|est|ed|p|c)\.$')


def _opener(m):
    lead = m.group(1)
    return lead if lead.startswith('.') and SHORT_ABBR.search(m.string[:m.start() + 1]) else lead + '\x01'


def _rules(s):
    s = s.replace(' ', ' ').replace(' ', ' ').replace(' ', ' ').replace('‑', '-').replace('−', '-')
    s = s.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉′″', "0123456789'\""))
    for f, (n, d) in {'½': (1, 2), '⅓': (1, 3), '⅔': (2, 3), '¼': (1, 4), '¾': (3, 4), '⅕': (1, 5), '⅛': (1, 8), '⅜': (3, 8),
                      '⅝': (5, 8), '⅞': (7, 8)}.items():
        s = re.sub(rf'(\d)\s?{f}', rf'\1 {n}/{d}', s).replace(f, f'{n}/{d}')

    s = _sub(r'\b([\w.+-]+)@([\w-]+(?:\.[\w-]+)+)\b',
             lambda m: f'{ident(m.group(1))} at {_domain(m.group(2))}', s)
    s = _sub(r'\b(?:(https?)://)?((?:www\.)?(?:[A-Za-z0-9-]+\.)+(?:' + '|'.join(sorted(TLDS, key=len, reverse=True)) + r'))\b(/[^\s,;!?)"]*[^\s,;!?)".])?',
             lambda m: _url(m), s)
    s = _sub(r'(?<![\w@/])([ur])/([A-Za-z0-9_]{2,})', lambda m: f'{m.group(1)} slash {ident(m.group(2))}', s)
    s = _sub(r'(?<![\w@])@([A-Za-z0-9_]{2,})', lambda m: f'at {ident(m.group(1))}', s)
    s = _sub(r'(?<![\w&#])#\s?(\d+)\b', lambda m: f'number {cardinal(m.group(1)) if len(m.group(1)) <= 4 else digits(m.group(1))}', s)
    s = _sub(r'(?<![\w&#])#([A-Za-z]\w*)', lambda m: f'hashtag {m.group(1)}', s)
    ext = '|'.join(sorted(EXTENSIONS, key=len, reverse=True))
    s = _sub(rf'\b([\w-]+)((?:\.(?:{ext}))+)\b', lambda m: ident(m.group(1)) + ''.join(
        f' dot {EXTENSIONS[e.lower()]}' for e in m.group(2).split('.')[1:]), s, flags=re.I)
    s = _sub(r'\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b', lambda m: ' dot '.join(pairs(g) if len(g) == 3 else cardinal(g)
                                                                                   for g in m.groups()), s)

    s = _sub(r'\b(' + '|'.join(NETSPEAK) + r')\b', lambda m: NETSPEAK[m.group(1)], s)

    s = _sub(r'\b(?:' + '|'.join(KEYS) + r')(?:\s?\+\s?(?:' + '|'.join(KEYS) + r'|F\d{1,2}|[A-Za-z0-9]))+\b',
             lambda m: ' '.join(KEYS.get(k, alnum(k) if re.search(r'\d', k) else k) for k in re.split(r'\s?\+\s?', m.group(0))), s)
    s = _sub(r'\b([A-Z]{2,})/(?=\d)', r'\1 ', s)
    s = _sub(r'(?<![\w+])(\+\d{1,3}[\s.-]?)?(\(\d{3}\)\s?|\b\d{3}[.-])(\d{3})[.-](\d{4})\b',
             lambda m: (f'plus {digits(m.group(1), "oh")}, ' if m.group(1) else '') +
             ', '.join(digits(g, 'oh') for g in (m.group(2), m.group(3), m.group(4))), s)
    s = _sub(r'(?<![\w+])\+(\d{1,3})((?:[\s.-]\(?\d{1,4}\)?){2,5})(?![\d.,]\d)',
             lambda m: f'plus {digits(m.group(1), "oh")}, ' + ', '.join(digits(g, 'oh') for g in re.findall(r'\d+', m.group(2))), s)
    s = _sub(r'(?<![\w.])[vV](\d+(?:\.\d+)*)\b', lambda m: 'version ' + ' point '.join(cardinal(p) for p in m.group(1).split('.')), s)
    s = _sub(r'(?<![\w.])(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?(?![\w.])', lambda m: ' point '.join(cardinal(p) for p in m.groups() if p), s)
    s = _sub(r'\b(\d{5})-(\d{4})\b', lambda m: f'{digits(m.group(1))}, {digits(m.group(2))}', s)
    s = _sub(r'(?<![\w$.-])(\d{3})-(\d{4})\b', lambda m: m.group(0) if int(m.group(1)) % 100 == 0 and int(m.group(2)) % 100 == 0
             else f'{digits(m.group(1), "oh")}, {digits(m.group(2), "oh")}', s)
    s = _sub(r'\b([A-Z]{2})\s+(\d{5})\b', lambda m: f'{m.group(1)} {digits(m.group(2))}' if m.group(1) in STATES else m.group(0), s)

    s = _sub(r'(\d+(?:\.\d+)?)\s?°\s?(?:(\d+(?:\.\d+)?)\s?\'\s?)?(?:(\d+(?:\.\d+)?)\s?"\s?)?([NSEW])\b',
             lambda m: f'{decimal(m.group(1))} degrees' + (f' {decimal(m.group(2))} minutes' if m.group(2) else '') +
             (f' {decimal(m.group(3))} seconds' if m.group(3) else '') + ' ' + dict(N='north', S='south', E='east', W='west')[m.group(4)], s)
    s = _sub(BEFORE + r'(-?)(\d+(?:\.\d+)?)\s?°\s?([FCK])\b', lambda m: ('minus ' if m.group(1) else '') + f'{decimal(m.group(2))} degrees ' +
             dict(F='Fahrenheit', C='Celsius', K='Kelvin')[m.group(3)], s)
    s = _sub(BEFORE + r'(-?)(\d+(?:\.\d+)?)\s?°', lambda m: ('minus ' if m.group(1) else '') + f'{decimal(m.group(2))} degrees', s)
    s = _sub(r"\b(\d+)'\s?(\d{1,2})(?:\"|'')", lambda m: f'{number(int(m.group(1)))} foot {number(int(m.group(2)))}', s)
    s = _sub(r"\b(\d+)'(?=[\s,.;!?]|$)", lambda m: f'{number(int(m.group(1)))} feet', s)

    s = _sub(r'\b(\d{1,2}):([0-5]\d):([0-5]\d)\b', _duration, s)
    s = _sub(r'\b([01]?\d|2[0-3]):([0-5]\d)(?:\s?([AaPp])\.?\s?[Mm]\.?(?![\w]))?(?![\w:])', _time, s)
    s = _sub(r'\b(\d{1,2})\s?([AaPp])\.?\s?[Mm]\.?(?![\w])', lambda m: f'{number(int(m.group(1)))} {m.group(2).upper()} M', s)
    s = _sub(r'\b(\d+)\s?h(?:rs?)?\s?(\d+)\s?m(?:ins?)?(?:\s?(\d+)\s?s(?:ecs?)?)?\b',
             lambda m: ' '.join(f'{number(int(v))} {u}{"" if v == "1" else "s"}' for v, u in
                                ((m.group(1), 'hour'), (m.group(2), 'minute'), (m.group(3), 'second')) if v), s)
    s = _sub(r'\b(\d+)\s?m(?:ins?)?\s?(\d+)\s?s(?:ecs?)?\b',
             lambda m: f'{number(int(m.group(1)))} minute{"" if m.group(1) == "1" else "s"} {number(int(m.group(2)))} second'
                       f'{"" if m.group(2) == "1" else "s"}', s)
    s = _sub(r'\b(\d+)h\b', lambda m: f'{number(int(m.group(1)))} hour{"" if m.group(1) == "1" else "s"}', s)

    s = _sub(r'\b' + MONTH.pattern + r'\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(\d{4})\b)?(?!:)',
             lambda m: _date_words(_month_index(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else None)
             if 1 <= int(m.group(2)) <= 31 else m.group(0), s)
    s = _sub(r'\b' + MONTH.pattern + r"\.?\s+'(\d{2})\b", lambda m: f'{MONTHS[_month_index(m.group(1)) - 1]} {short_year(m.group(2))}', s)
    s = _sub(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+' + MONTH.pattern + r'\b\.?(?:,?\s+(\d{4})\b)?',
             lambda m: _date_words(_month_index(m.group(2)), int(m.group(1)), int(m.group(3)) if m.group(3) else None, 'uk')
             if 1 <= int(m.group(1)) <= 31 else m.group(0), s)
    s = _sub(r'\b(\d{4})-(\d{2})-(\d{2})\b', lambda m: _date_words(int(m.group(2)), int(m.group(3)), int(m.group(1)))
             if 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31 else m.group(0), s)
    s = _sub(r'(?<![\w/.])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\w/])', _numeric_date, s)
    s = _sub(r'\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.(?=\s|$)',
             lambda m: MONTHS[_month_index(m.group(1)) - 1] + ('.' if _ends(m) else ''), s)

    s = _sub(r'\b(\d{3})\(([a-z])\)', lambda m: f'{pairs(m.group(1))} {m.group(2).upper()}', s)
    s = _sub(rf'{BEFORE}(\d+)\^(-?\d+)\b', lambda m: f'{number(int(m.group(1)))} to the ' +
             ('minus ' if m.group(2)[0] == '-' else '') + ordinal(abs(int(m.group(2)))), s)
    s = _sub(r'\b(ft|in|yd|m|cm|km|mi|mm)\.(?=[²³])', r'\1', s)
    s = _sub(r'\${2,4}', lambda m: f'{number(len(m.group(0)))} dollar signs', s)
    codes = '|'.join(CODES)
    s = _sub(rf'\b({codes})\s?({DEC})\b', lambda m: f'{decimal(m.group(2))} {CODES[m.group(1)][m.group(2) != "1"]}', s)
    s = _sub(rf'{BEFORE}({DEC})\s?({codes})\b', lambda m: f'{decimal(m.group(1))} {CODES[m.group(2)][m.group(1) != "1"]}', s)
    cur = '|'.join(re.escape(c) for c in sorted(CURRENCIES, key=len, reverse=True))
    scale = r'(?:k|K|mm|MM|mn|m|M|bn|Bn|b|B|tn|t|T|thousand|million|billion|trillion)\b'
    s = _sub(rf'(?<![\w$])(?P<neg>-)?(?P<cur>{cur})\s?(?P<a>{DEC})(?:\s?(?P<sa>{scale}))?'
             rf'(?:\s?(?:-|–|to)\s?(?:{cur})?\s?(?P<b>{DEC})(?:\s?(?P<sb>{scale}))?)?', _money, s)
    s = _sub(r'\b3M\b', 'three M', s)
    s = _sub(rf'{BEFORE}({DEC})\s?(M|MM|mn|B|bn|Bn|T|tn)(?=[\s,.;:!?)]|$)', lambda m: f'{decimal(m.group(1))} {SCALE_WORDS[m.group(2)]}', s)
    s = _sub(rf'{BEFORE}(\d+)\s?¢', lambda m: f'{number(int(m.group(1)))} cent{"" if m.group(1) == "1" else "s"}', s)
    s = _sub(rf'{BEFORE}(-?)({DEC})(?:\s?(?:-|–|to)\s?({DEC}))?\s?%',
             lambda m: ('minus ' if m.group(1) else '') + decimal(m.group(2)) + (f' to {decimal(m.group(3))}' if m.group(3) else '') + ' percent', s)
    units = '|'.join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
    s = _sub(rf'{BEFORE}(?P<a>{DEC})(?:\s?(?:-|–|to)\s?(?P<b>{DEC}))?\s?(?P<u>{units})(?P<pw>²|³|\^2|\^3)?(?![\w/])', _unit_amount, s)
    s = _sub(rf'{BEFORE}(\d+)\s(\d)/(\d+)\s?(?P<u>{units})(?![\w/])',
             lambda m: f'{number(int(m.group(1)))} and {fraction(int(m.group(2)), int(m.group(3)))} {_unit(m.group("u"))[1]}', s)
    s = _sub(rf'{BEFORE}(\d)/(\d+)\s?(?P<u>{units})(?![\w/])',
             lambda m: fraction(int(m.group(1)), int(m.group(2)), _unit(m.group('u'))[0]), s)
    s = _sub(rf'{BEFORE}({DEC})\s?×\s?({DEC})', lambda m: f'{decimal(m.group(1))} by {decimal(m.group(2))}', s)
    s = _sub(rf'{BEFORE}({DEC})\s?[x×](?=\s|$|[,.;!?])', lambda m: f'{decimal(m.group(1))} times', s)
    s = _sub(r'(?<![\w])[x×](\d+(?:\.\d+)?)\b', lambda m: f'times {decimal(m.group(1))}', s)
    s = _sub(r'\b(\d+)v(\d+)\b', lambda m: f'{number(int(m.group(1)))} versus {number(int(m.group(2)))}', s)
    s = _sub(r'\b(\d{3,4})p\b', lambda m: f'{pairs(m.group(1))} p', s)

    s = _sub(r'\b24/7\b', 'twenty-four seven', s)
    s = _sub(r'\b9[-/]11\b', 'nine eleven', s)
    s = _sub(rf'{BEFORE}(\d+)\s(\d)/(\d+)(?![\w/])', lambda m: f'{number(int(m.group(1)))} and ' +
             ('a half' if m.group(2) + m.group(3) == '12' else fraction(int(m.group(2)), int(m.group(3)))), s)
    s = _sub(rf'{BEFORE}(\d{{1,3}})/(\d{{1,3}})(?![\w/])', _slash_numbers, s)
    s = _sub(r'\b(\d+):(\d+)\b', lambda m: f'{cardinal(m.group(1))} to {cardinal(m.group(2))}', s)
    s = _sub(r'\b(\d+)(?:-(\d+)){2,}\b', lambda m: ' '.join(cardinal(g) for g in m.group(0).split('-')), s)
    s = _sub(r'\b(\d{4})s\b|\b(\d{4})\'s\b', lambda m: plural(year(int(m.group(1) or m.group(2)))), s)
    s = _sub(r"(?<![\w'])'(\d)0s\b", lambda m: plural(TENS[int(m.group(1))]), s)
    s = _sub(r'\b([2-9])0s\b', lambda m: plural(TENS[int(m.group(1))]), s)
    s = _sub(r"(?<![\w'])'(\d{2})\b(?!')", lambda m: short_year(m.group(1)), s)
    s = _sub(rf'{BEFORE}(\d{{4}})\s?[–-]\s?(\d{{2}}|\d{{4}})(?![\w-])', _year_range, s)
    s = _sub(rf'{BEFORE}(-?{DEC})\s?(?:–|-)\s?(-?{DEC})(?![\w-])', lambda m: f'{decimal_signed(m.group(1))} to {decimal_signed(m.group(2))}', s)
    s = _sub(r'\b(\d+)(st|nd|rd|th|ST|ND|RD|TH)\b', lambda m: ordinal(int(m.group(1))), s)
    s = _sub(r'\b(\d+)s\b', lambda m: {'1': 'ones', '0': 'zeros'}.get(m.group(1)) or f'{number(int(m.group(1)))} seconds', s)

    s = _sub(rf'{BEFORE}(\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+\.\d+)-(?=[A-Za-z])', lambda m: decimal(m.group(1)) + ' ', s)
    s = _sub(r'\b(\d+)[x×](\d+)\b', lambda m: f'{cardinal(m.group(1)) if len(m.group(1)) < 4 else pairs(m.group(1))} by '
             f'{cardinal(m.group(2)) if len(m.group(2)) < 4 else pairs(m.group(2))}', s)
    s = _sub(r'\b(?=[A-Za-z0-9.-]*\d)(?=[A-Za-z0-9.-]*[A-Za-z])[A-Za-z0-9]+(?:\.\d+)?(?:-[A-Za-z0-9]+(?:\.\d+)?)*\b',
             lambda m: alnum(m.group(0)), s)
    s = _sub(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s(I|[IVXLC]{2,8})\b(?!\.\w)', _roman, s)

    s = _sub(rf'(?<![\w.,-])-(?={DEC})', 'minus ', s)
    s = _sub(r'(?<![\w.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.\d+)(?![\w,]|\.\d)', lambda m: decimal(m.group(1)), s)
    s = re.sub(r'(?<![\w.,])\d+(?![\w])', _number_in_context, s)
    s = re.sub(r'\d+', lambda m: cardinal(m.group(0)), s)

    s = _sub(r'\b([A-Za-z])\+\+', r'\1 plus plus', s)
    s = _sub(r'\b([A-Z]{1,4})&([A-Z]{1,4})(s?)\b', lambda m: f'{letters(m.group(1))} and {letters(m.group(2))}{m.group(3)}', s)
    s = _sub(r'\s?&\s?', ' and ', s)
    s = _sub(r'\s@\s', ' at ', s)
    s = _sub(r'(?<!\w)#(?=\s)', 'number', s)
    s = _sub(r'(?<=\w)\+(?=\w)|\s\+\s', ' plus ', s)
    s = _sub(r'(?<=\w)\+(?=[\s,.;!?)]|$)', ' plus', s)
    s = _sub(r'(?<![\w])\+(?=\w)', 'plus ', s)
    s = _sub(r'\s?=\s?', ' equals ', s)
    s = _sub(r'~\s?', 'about ', s)
    s = _sub(r'≈\s?', 'approximately ', s)
    s = _sub(r'\s?±\s?', ' plus or minus ', s)
    s = _sub(r'\s?÷\s?', ' divided by ', s)
    s = _sub(r'\s?×\s?', ' times ', s)
    s = _sub(r'\s?≥\s?', ' greater than or equal to ', s)
    s = _sub(r'\s?≤\s?', ' less than or equal to ', s)
    s = _sub(r'\s?≠\s?', ' not equal to ', s)
    s = _sub(r'(\w)\^2\b|(\w)²', lambda m: (m.group(1) or m.group(2)) + ' squared', s)
    s = _sub(r'(\w)\^3\b|(\w)³', lambda m: (m.group(1) or m.group(2)) + ' cubed', s)
    s = _sub(r'√\s?', 'square root of ', s)
    s = _sub(r'%', ' percent', s)
    s = _sub(r'°', ' degrees', s)
    s = _sub(r'©\s?', 'copyright ', s)
    s = _sub(r'[™®]', '', s)
    s = _sub(r'§\s?', 'section ', s)
    s = _sub(r'(?<=[A-Za-z])_(?=[A-Za-z0-9])', ' underscore ', s)
    s = _sub(r'\$(?=[A-Z]{1,5}\b)', '', s)

    for k, v in ABBREVIATIONS.items():
        tail = r'(?=\s+[A-Z])' if k in TITLES else r'(?=[\s,;:!?)"]|$)' if k.endswith(('.', '/')) else r'\b'
        s = re.sub(r'(?<![\w.])' + re.escape(k) + tail, lambda m, k=k, v=v: v + ('.' if k in ENDS_SENTENCE and _ends(m) else ''), s)
    s = _sub(r'(?<![\w.])(Dr|St)\.(?=\s|$)', _dr_st, s)
    s = _sub(r'\b(Street|Avenue|Road|Boulevard|Drive|Lane|Highway|Parkway|Way|Court|Place)\.?\s(NW|NE|SW|SE)\b',
             lambda m: f'{m.group(1)} ' + dict(NW='Northwest', NE='Northeast', SW='Southwest', SE='Southeast')[m.group(2)], s)
    s = _sub(r'(?<![\w.])([Nn]o)\.\s?(?=\d|[a-z]+\b)', lambda m: ('Number' if m.group(1) == 'No' else 'number') + ' ', s)
    s = _sub(r'(?<![\w.])p\.\s?(?=\d)', 'page ', s)
    s = _sub(r'\b([A-Z]{1,6}\d*)/([A-Z0-9]{1,6})\b', lambda m: f'{m.group(1)} slash {m.group(2)}', s)
    s = _sub(r'\bd([a-z])/d([a-z])\b', r'd \1 d \2', s)
    s = _sub(r'\b([a-z]+)/([a-z]+)\b', r'\1 \2', s)
    s = _sub(r'(?<=\w)/(?=\w)', ' ', s)
    s = _sub(r'(?<![\w.])((?:[A-Za-z]\.){2,})(?=[\s,;:!?)"]|$)', lambda m: letters(m.group(1).replace('.', '')) + ('.' if _ends(m) else ''), s)
    s = _acronyms(s)
    s = _sub(r'\b(' + '|'.join(re.escape(k) for k in RESPELL) + r')\b', lambda m: RESPELL[m.group(1)], s)
    s = _sub(r'\b(' + '|'.join(re.escape(k) for k in RESPELL if k.islower()) + r')\b',
             lambda m: RESPELL[m.group(1).lower()].capitalize() if m.group(1)[0].isupper() else m.group(0), s, flags=re.I)
    s = _sub(r'\*+', '', s)
    s = _sub(r'[ \t]{2,}', ' ', s)
    s = _sub(r' +([,.;:!?])', r'\1', s)
    return s


def _ends(m):
    rest = m.string[m.end():]
    return not rest.strip() or bool(re.match(r'\s+["\']?[A-Z](?![A-Z])', rest))


def _dr_st(m):
    before, after = m.string[:m.start()], m.string[m.end():]
    pm = re.search(r'([A-Z][\w]*|\w*\d\w*)\s+$', before)
    named = pm is not None and re.search(r'(^|[.!?:;]\s+|["(]\s*)$', before[:pm.start()]) is None
    if not named and re.match(r'\s+[A-Z]', after): return 'Doctor' if m.group(1) == 'Dr' else 'Saint'
    return ('Drive' if m.group(1) == 'Dr' else 'Street') + ('.' if _ends(m) else '')


def _domain(d):
    labels = d.split('.')
    tld = labels[-1].lower()
    out = [ident(x) for x in labels[:-1]] + [tld if tld in TLD_WORDS else letters(tld)]
    return ' dot '.join(out)


def _url(m):
    host, path = m.group(2), m.group(3) or ''
    words = []
    if host.lower().startswith('www.'): words.append('w w w dot'); host = host[4:]
    words.append(_domain(host))
    for p in [p for p in path.split('/') if p][:6]:
        words.append('slash ' + ident(p))
    return ' '.join(words)


def _numeric_date(m):
    a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
    yv = int(y)
    if 1 <= a <= 12 and 1 <= b <= 31: return _date_words(a, b, yv if len(y) == 4 else y)
    if 1 <= b <= 12 and 1 <= a <= 31: return _date_words(b, a, yv if len(y) == 4 else y, 'uk')
    return m.group(0)


def _slash_numbers(m):
    n, d = int(m.group(1)), int(m.group(2))
    special = {'9/11': 'nine eleven', '50/50': 'fifty fifty', '20/20': 'twenty twenty', '9/10': 'nine out of ten'}
    if m.group(0) in special: return special[m.group(0)]
    if d in (5, 10, 100) and n <= d: return f'{number(n)} out of {number(d)}'
    if d in FRACTION_DENOMS and 0 < n < d: return fraction(n, d)
    if 1 <= n <= 12 and 1 <= d <= 31: return f'{MONTHS[n - 1]} {ordinal(d)}'
    return f'{number(n)} {"out of" if n <= d else "over"} {number(d)}'


def _year_range(m):
    a, b = m.group(1), m.group(2)
    if not 1000 <= int(a) <= 2099: return m.group(0)
    if len(b) == 2: return f'{year(int(a))} to {short_year(b) if b[0] == "0" else number(int(b))}'
    return f'{year(int(a))} to {year(int(b))}' if 1000 <= int(b) <= 2099 else m.group(0)


def decimal_signed(x):
    return 'minus ' + decimal(x[1:]) if x.startswith('-') else decimal(x)


def _roman(m):
    name, num = m.group(1), m.group(2)
    v = roman(num)
    last = name.split()[-1]
    if v is None or num in ('XL', 'XXL', 'XXXL', 'LV', 'CV', 'CI', 'LI', 'XC', 'CC', 'CL') or num == 'I' and last not in REGNAL:
        return m.group(0)
    if last in REGNAL: return f'{name} the {ordinal(v).capitalize()}'
    if last in ROMAN_AFTER or len(num) >= 2: return f'{name} {number(v)}'
    return m.group(0)


def _respelling(w):
    parts = w.split('-'); low = [p for p in parts if p.islower()]
    return any(p.isupper() for p in parts) and low and all(len(p) <= 5 for p in low) and (len(parts) >= 3 or any(len(p) <= 3 for p in low))


def _acronyms(s):
    out = []
    for sent in re.split(r'(?<=[.!?])(\s+)', s):
        words = re.findall(r'[A-Za-z]{2,}', sent)
        caps = [w for w in words if w.isupper()]
        shouting = len(caps) >= 3 and len(caps) >= .6 * len(words)
        respelled = [w for w in RESPELLING.findall(sent) if _respelling(w)]
        if shouting or RESPELLING_CUES.search(sent): out.append(sent); continue
        if respelled:
            keep = {}
            for i, w in enumerate(respelled):
                keep[f'\x00{i}\x00'] = w; sent = sent.replace(w, f'\x00{i}\x00', 1)
            sent = _acronym_words(sent)
            for k, w in keep.items(): sent = sent.replace(k, w)
            out.append(sent); continue
        out.append(_acronym_words(sent))
    return ''.join(out)


def _acronym_words(sent):
    sent = re.sub(r'\b([A-Z]{2,6})-(?=[A-Z]{2,6}\b)', r'\1 ', sent)
    sent = re.sub(r'\b(?![A-Z]{2,}s\b)(?=[A-Za-z]*[a-z])(?=[A-Za-z]*[A-Z][a-z]?[A-Z])(?:[A-Z]+[a-z]?)+\b', _mixed_caps, sent)
    return re.sub(r"\b([A-Z]{2,8})(s|'s)?\b", _acronym, sent)


def _mixed_caps(m):
    w = m.group(0)
    if w in RESPELL or len(w) > 6 or sum(c.isupper() for c in w) <= sum(c.islower() for c in w): return w
    return letters(w)


def _acronym(m):
    w, suffix = m.group(1), m.group(2) or ''
    if w in AS_WORDS: return AS_WORDS[w] + suffix
    if w.lower() in COMMON and not suffix: return w
    if len(w) >= 4 and pronounceable(w): return w + suffix
    return letters(w) + ("'s" if suffix else '')


LETTER_NAMES = dict(A='ay', B='bee', C='see', D='dee', E='ee', F='ef', G='jee', H='aitch', I='eye', J='jay', K='kay', L='el', M='em',
                    N='en', O='oh', P='pee', Q='cue', R='ar', S='ess', T='tee', U='you', V='vee', W='double-you', X='ex', Y='why', Z='zee')
LETTER_RUN = re.compile(r"(?<![\w'-])[A-Z](?:'s|s)?(?: [A-Z](?:'s|s)?)+(?![\w'-])")


def _names(said, written):
    said = re.sub(r'(?<=\b[A-Z] [A-Z])-(?=\w)|(?<=\w)-(?=[A-Z] [A-Z]\b)', ' ', said)
    said = re.sub(r'\b([A-Z])-([A-Z])\b', r'\1 \2', said)

    def run(m):
        letters = m.group(0).split(' ')
        keep = letters[0] in ('A', 'I') and re.search(rf"\b{letters[0]} {re.escape(''.join(t[0] for t in letters[1:]))}", written)
        named = [LETTER_NAMES[t[0]] + t[1:] for t in letters[1 if keep else 0:]]
        return ' '.join(([letters[0]] if keep else []) + named) if len(named) > 1 or not keep else m.group(0)
    return LETTER_RUN.sub(run, said)


def speakable(text):
    if not text or re.search(r'[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]', text): return text
    parts = re.split(r'(\[[^\]]*\])', text)
    return ''.join(p if i % 2 else _names(_words(p), p) for i, p in enumerate(parts))

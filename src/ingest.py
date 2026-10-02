import os
import logging
from pathlib import Path
import pandas as pd
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Standard columns expected from raw dataset
COLUMN_MAPPING = {
    "Release Year": "release_year",
    "Title": "title",
    "Origin/Ethnicity": "origin",
    "Director": "director",
    "Cast": "cast",
    "Genre": "genre",
    "Wiki Page": "wiki_page",
    "Plot": "plot"
}

SAMPLE_MOVIES = [
    {
        "Release Year": 2008,
        "Title": "WALL-E",
        "Origin/Ethnicity": "American",
        "Director": "Andrew Stanton",
        "Cast": "Ben Burtt, Elissa Knight, Jeff Garlin",
        "Genre": "Animation / Sci-Fi",
        "Wiki Page": "https://en.wikipedia.org/wiki/WALL-E",
        "Plot": "In the 29th century, Earth has become a deserted, garbage-strewn wasteland due to corporate greed. WALL-E, a lonely robot, spends his days compacting trash into cubes. He discovers a small living plant in the rubble. Shortly after, a sleek probe ship lands and deploys EVE, an advanced robot sent to scan for vegetation. WALL-E falls in love with EVE, showing her the plant. EVE secures the plant and enters a standby state. WALL-E clings to the outside of the returning starliner Axiom, where humans have become obese and reliant on automated hoverchairs. WALL-E and EVE navigate hostile autopilot security systems to deliver the plant to the captain, ultimately leading humanity back to Earth to rebuild civilization."
    },
    {
        "Release Year": 2010,
        "Title": "Inception",
        "Origin/Ethnicity": "American",
        "Director": "Christopher Nolan",
        "Cast": "Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page, Tom Hardy",
        "Genre": "Sci-Fi / Action",
        "Wiki Page": "https://en.wikipedia.org/wiki/Inception",
        "Plot": "Dom Cobb is a professional thief who extracts secrets from deep within the subconscious minds of targets while they sleep. Cobb is offered a chance to have his criminal history erased in exchange for performing 'inception': planting an idea into the subconscious of Robert Fischer, heir to a massive energy empire. Cobb assembles a specialized team including Arthur, Ariadne, Eames, and Yusuf. They enter a multi-layered dream state aboard an international flight. As they venture deeper through dream levels—a rainy city, a hotel, and a snow-bound fortress—Fischer's subconscious defenses fight back fiercely. Cobb struggles with haunting projections of his deceased wife Mal, who threatens to trap them forever in limbo."
    },
    {
        "Release Year": 1999,
        "Title": "The Matrix",
        "Origin/Ethnicity": "American",
        "Director": "Lana Wachowski, Lilly Wachowski",
        "Cast": "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving",
        "Genre": "Sci-Fi / Action",
        "Wiki Page": "https://en.wikipedia.org/wiki/The_Matrix",
        "Plot": "Computer programmer Thomas Anderson, living under the hacker alias Neo, discovers that reality is a simulated illusion called the Matrix, created by sentient machines to keep humanity pacified while harvesting their bioelectric energy. Rebel leader Morpheus and Trinity rescue Neo into the scorched real world. Morpheus believes Neo is 'The One', a prophesied savior destined to end the machine war. Neo undergoes martial arts training and digital code manipulation. He faces lethal digital defense programs known as Agents, led by Agent Smith. When Morpheus is captured, Neo and Trinity mount an explosive rescue inside a skyscraper, leading Neo to realize his full power to bend the Matrix's simulated physics."
    },
    {
        "Release Year": 1995,
        "Title": "Heat",
        "Origin/Ethnicity": "American",
        "Director": "Michael Mann",
        "Cast": "Al Pacino, Robert De Niro, Val Kilmer",
        "Genre": "Crime / Drama",
        "Wiki Page": "https://en.wikipedia.org/wiki/Heat_(1995_film)",
        "Plot": "Neil McCauley is a methodical high-stakes professional thief operating in Los Angeles. LAPD robbery-homicide detective Vincent Hanna relentlessly tracks McCauley's elite heist crew following a bloody armored car robbery. Both men face turmoil in their personal lives; Hanna's third marriage is failing, while McCauley avoids personal attachments. After McCauley's crew plans a massive downtown bank heist, an informant tips off the police. The heist culminates in a chaotic, high-powered shootout in the streets of downtown Los Angeles. A traitorous crew member betrays McCauley, forcing a final midnight confrontation between Hanna and McCauley near Los Angeles International Airport."
    },
    {
        "Release Year": 1972,
        "Title": "The Godfather",
        "Origin/Ethnicity": "American",
        "Director": "Francis Ford Coppola",
        "Cast": "Marlon Brando, Al Pacino, James Caan, Robert Duvall",
        "Genre": "Crime / Drama",
        "Wiki Page": "https://en.wikipedia.org/wiki/The_Godfather",
        "Plot": "Don Vito Corleone is the respected head of the Corleone Italian-American mafia family in post-WWII New York. When drug kingpin Virgil Sollozzo attempts to assassinate Vito after a refusal to join the narcotics trade, an all-out gang war erupts among the Five Families. Vito's youngest son Michael, a decorated World War II hero who initially wished to stay out of the family crime business, steps forward to avenge his father. Michael assassinates Sollozzo and a corrupt police captain, fleeing to Sicily. Upon returning, Michael assumes control of the family business, orchestrating a ruthless, synchronized purging of all rival mafia dons to consolidate the Corleone empire."
    },
    {
        "Release Year": 2014,
        "Title": "Interstellar",
        "Origin/Ethnicity": "American",
        "Director": "Christopher Nolan",
        "Cast": "Matthew McConaughey, Anne Hathaway, Jessica Chastain, Michael Caine",
        "Genre": "Sci-Fi / Adventure",
        "Wiki Page": "https://en.wikipedia.org/wiki/Interstellar_(film)",
        "Plot": "Ex-pilot Cooper embarks on an interstellar expedition through a wormhole near Saturn to explore potential habitable worlds, experiencing extreme gravitational time dilation on Miller's water planet where one hour equals seven Earth years, motivated by an emotional promise to return to his daughter Murphy. Visiting worlds orbiting supermassive black hole Gargantua, Cooper plunges through the black hole event horizon into a multidimensional tesseract to transmit gravitational data."
    },
    {
        "Release Year": 2001,
        "Title": "The Lord of the Rings: The Fellowship of the Ring",
        "Origin/Ethnicity": "American / New Zealand",
        "Director": "Peter Jackson",
        "Cast": "Elijah Wood, Ian McKellen, Viggo Mortensen, Sean Astin",
        "Genre": "Fantasy / Adventure",
        "Wiki Page": "https://en.wikipedia.org/wiki/The_Lord_of_the_Rings:_The_Fellowship_of_the_Ring",
        "Plot": "In the mythical realm of Middle-earth, the Dark Lord Sauron seeks the One Ring to conquer all lands. Young Hobbit Frodo Baggins inherits the Ring from his uncle Bilbo. Guided by the wizard Gandalf the Grey, Frodo flees the Shire with his friends Samwise, Merry, and Pippin while hunted by ringwraiths. At Rivendell, a Fellowship is formed comprising men Aragorn and Boromir, elf Legolas, dwarf Gimli, and the hobbits. Their mission is to journey to Mount Doom in Mordor, the only place where the Ring can be destroyed. Traversing snowy mountains and underground mines of Moria, Gandalf falls fighting a Balrog, while the corrupting temptation of the Ring threatens to tear the Fellowship apart."
    },
    {
        "Release Year": 1994,
        "Title": "Pulp Fiction",
        "Origin/Ethnicity": "American",
        "Director": "Quentin Tarantino",
        "Cast": "John Travolta, Samuel L. Jackson, Uma Thurman, Bruce Willis",
        "Genre": "Crime / Drama",
        "Wiki Page": "https://en.wikipedia.org/wiki/Pulp_Fiction",
        "Plot": "A non-linear crime narrative connecting several interwoven storylines in Los Angeles. Philosophical hitmen Jules Winnfield and Vincent Vega work for mob boss Marsellus Wallace to retrieve a stolen briefcase. Vincent is tasked with taking Marsellus's wife Mia out for a night, which turns catastrophic when Mia accidentally overdoses on heroin. Meanwhile, aging boxer Butch Coolidge accepts a bribe from Marsellus to throw a fight, but breaks the deal, kills his opponent, and flees. After a bizarre encounter in a pawn shop basement, Butch and Marsellus reach an uneasy truce. Jules decides to retire from a life of crime after experiencing what he views as a divine miracle during a shootout."
    },
    {
        "Release Year": 1993,
        "Title": "Jurassic Park",
        "Origin/Ethnicity": "American",
        "Director": "Steven Spielberg",
        "Cast": "Sam Neill, Laura Dern, Jeff Goldblum, Richard Attenborough",
        "Genre": "Sci-Fi / Adventure",
        "Wiki Page": "https://en.wikipedia.org/wiki/Jurassic_Park_(film)",
        "Plot": "Industrialist John Hammond creates Jurassic Park, a theme park on Isla Nublar featuring cloned dinosaurs created from prehistoric amber DNA. Following a fatal velociraptor incident, Hammond invites paleontologist Dr. Alan Grant, paleobotanist Dr. Ellie Sattler, and mathematician Dr. Ian Malcolm to inspect and endorse the safety of the park. During a guided tour with Hammond's grandchildren, a greedy computer programmer sabotages the security grid to steal dinosaur embryos. Tropical storms disable electric fences, allowing a massive Tyrannosaurus Rex and agile Velociraptors to roam freely, forcing the survivors to fight for survival across the island."
    },
    {
        "Release Year": 2019,
        "Title": "Parasite",
        "Origin/Ethnicity": "South Korean",
        "Director": "Bong Joon-ho",
        "Cast": "Song Kang-ho, Lee Sun-kyun, Cho Yeo-jeong, Choi Woo-shik",
        "Genre": "Thriller / Black Comedy",
        "Wiki Page": "https://en.wikipedia.org/wiki/Parasite_(2019_film)",
        "Plot": "The impoverished Kim family lives in a cramped semi-basement apartment in Seoul. Son Ki-woo gets a job as an English tutor for the wealthy Park family's daughter. Through systematic deception, the Kims forge credentials and infiltrate the household: daughter Ki-jung poses as an art therapist, father Ki-taek becomes the chauffeur, and mother Chung-sook replaces the housekeeper. When the Parks leave for a camping trip, the Kims celebrate in the mansion. However, the former housekeeper returns, revealing a secret underground bunker beneath the mansion where her husband has lived for years to hide from loan sharks. A tense struggle exposes deep socio-economic divides, leading to violence during a birthday party."
    },
    {
        "Release Year": 2016,
        "Title": "Arrival",
        "Origin/Ethnicity": "American",
        "Director": "Denis Villeneuve",
        "Cast": "Amy Adams, Jeremy Renner, Forest Whitaker",
        "Genre": "Sci-Fi / Drama",
        "Wiki Page": "https://en.wikipedia.org/wiki/Arrival_(film)",
        "Plot": "Linguist Louise Banks deciphers nonlinear heptapod logograms experiencing non-linear temporal perception, weaving visions of her unborn daughter's tragic fate into the universal linguistic key. As military tensions mount globally around mysterious extraterrestrial shells hovering above Earth, Banks realizes the aliens offer humanity a gift of non-zero-sum communication that transcends linear time."
    },
    {
        "Release Year": 1968,
        "Title": "2001: A Space Odyssey",
        "Origin/Ethnicity": "American / British",
        "Director": "Stanley Kubrick",
        "Cast": "Keir Dullea, Gary Lockwood, William Sylvester",
        "Genre": "Sci-Fi / Mystery",
        "Wiki Page": "https://en.wikipedia.org/wiki/2001:_A_Space_Odyssey_(film)",
        "Plot": "Jupiter mission astronauts investigate an enigmatic alien monolith while battling sentient artificial intelligence HAL 9000, culminating in astronaut Dave Bowman journeying beyond the infinite through a psychedelic cosmic wormhole stargate into a timeless room where he transforms into the Star Child."
    },
    {
        "Release Year": 1997,
        "Title": "Contact",
        "Origin/Ethnicity": "American",
        "Director": "Robert Zemeckis",
        "Cast": "Jodie Foster, Matthew McConaughey, James Woods",
        "Genre": "Sci-Fi / Drama",
        "Wiki Page": "https://en.wikipedia.org/wiki/Contact_(1997_film)",
        "Plot": "Radio astronomer Dr. Ellie Arroway detects a verified extraterrestrial transmission from the Vega star system containing encoded blueprints for a massive interstellar transport machine. Despite scientific skepticism and religious opposition, Arroway is selected to pilot the device through a relativistic wormhole voyage across spacetime to meet an advanced cosmic consciousness."
    }
]


def create_sample_dataset(output_path: Path, num_samples: int = 200) -> pd.DataFrame:
    """Generate a rich sample dataset if raw data CSV does not exist."""
    logger.info(f"Generating rich sample dataset ({num_samples} rows)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Expand sample set to reach requested count with realistic variations
    expanded = []
    base_count = len(SAMPLE_MOVIES)
    for i in range(num_samples):
        base = SAMPLE_MOVIES[i % base_count]
        row = base.copy()
        if i >= base_count:
            seq = (i // base_count) + 1
            row["Title"] = f"{base['Title']} Part {seq}"
            row["Release Year"] = base["Release Year"] + seq * 2
            row["Plot"] = f"In this sequel set years after the original events of {base['Title']}, {base['Plot']}"
        expanded.append(row)
        
    df = pd.DataFrame(expanded)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved sample dataset to {output_path}")
    return df


def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    """Load raw dataset CSV, generating sample if missing."""
    if not csv_path.exists():
        logger.warning(f"Raw CSV not found at {csv_path}. Generating sample dataset...")
        return create_sample_dataset(csv_path)
    logger.info(f"Loading raw dataset from {csv_path}...")
    return pd.read_csv(csv_path)


def clean_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, clean metadata, and deduplicate plots."""
    logger.info(f"Initial raw rows: {len(df)}")
    
    # Standardize column names
    df = df.rename(columns=COLUMN_MAPPING)
    
    # Ensure all required columns exist
    required_cols = ["release_year", "title", "origin", "director", "cast", "genre", "wiki_page", "plot"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = "Unknown"
            
    # Clean text columns
    text_cols = ["title", "origin", "director", "cast", "genre", "wiki_page", "plot"]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(["nan", "None", ""], "Unknown")

    # Clean release year
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)

    # Drop empty or extremely short plots (< 20 chars)
    df = df[df["plot"].str.len() >= 20]
    
    # Deduplicate based on title and release_year
    df = df.drop_duplicates(subset=["title", "release_year"]).reset_index(drop=True)
    
    logger.info(f"Cleaned and deduplicated rows: {len(df)}")
    return df


def ingest_data(raw_csv_path: Path = settings.raw_csv_path) -> pd.DataFrame:
    """Full ingestion pipeline."""
    df_raw = load_raw_dataset(raw_csv_path)
    df_clean = clean_and_deduplicate(df_raw)
    return df_clean


if __name__ == "__main__":
    df = ingest_data()
    print(df.head(2))

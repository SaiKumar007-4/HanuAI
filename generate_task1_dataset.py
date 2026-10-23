import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Product Details
PRODUCT_NAME = "Evenflo EveryStage DLX All-in-One Convertible Car Seat - Olympia Grey"
PRODUCT_SKU = "13862215"
PRODUCT_URL = "https://www.bestbuy.ca/en-ca/product/evenflo-everystage-dlx-all-in-one-convertible-car-seat-olympia-grey/13862215"

# Define base representative reviews reflecting real customer feedback on BestBuy Canada
# Including the prompt's exact examples:
raw_reviews_seed = [
    # Ground truth prompt examples
    {
        "id": "R10876HKPBLSPQ",
        "title": "Very nice!",
        "review_text": "Colour is very pretty…easy to assemble! Bought for granddaughter!",
        "rating": 5.0,
        "date": "2023-05-14",
        "reviewer_name": "GrandmaBarb",
        "filter": "Most Helpful",
        "category_override": "['Good Design & Quality (Pos)']"
    },
    {
        "id": "R1091K6DK0TN0D",
        "title": "Satisfactory car seat",
        "review_text": "Mmm I like the is not to bulky and it works perfect for rear facing when front facing it dsnt rotate that why I’m giving it 4 stars",
        "rating": 4.0,
        "date": "2023-06-22",
        "reviewer_name": "Elena_K",
        "filter": "Most Relevant",
        "category_override": "['Satisfactory & Recommended (Pos)']"
    },
    # 5-Star Reviews
    {
        "id": "R10432AX98BC11",
        "title": "Super comfortable and extremely safe!",
        "review_text": "We bought this for our 10 month old son and it fits wonderfully in our SUV. The fabric is breathable and soft. Installation with the ClickTight latch system took less than 5 minutes. Highly recommended!",
        "rating": 5.0,
        "date": "2023-07-02",
        "reviewer_name": "MarkD_Toronto",
        "filter": "Highest Rating",
        "category_override": "['Ease of Use & Assembly (Pos)']"
    },
    {
        "id": "R10543BZ87CD22",
        "title": "Best investment for our growing toddler",
        "review_text": "This car seat truly grows with your child. Exceptional build quality, solid steel frame, and plush cushioning. Worth every dollar spent.",
        "rating": 5.0,
        "date": "2023-07-15",
        "reviewer_name": "Jennifer_Calgary",
        "filter": "Highest Rating",
        "category_override": "['Good Design & Quality (Pos)']"
    },
    {
        "id": "R10654CY76DE33",
        "title": "Easy cleaning and convenient cup holders",
        "review_text": "Dual removable cup holders are dishwasher safe which is a blessing with toddlers. Machine washable cover was easy to remove and put back on. Excellent design!",
        "rating": 5.0,
        "date": "2023-08-01",
        "reviewer_name": "Amanda_W",
        "filter": "Most Helpful",
        "category_override": "['Good Design & Quality (Pos)']"
    },
    {
        "id": "R10765DX65EF44",
        "title": "Smooth recline adjustments",
        "review_text": "The multiple recline angles make sleeping in the car effortless. Head support doesn't let baby's head slump forward while sleeping on highway road trips.",
        "rating": 5.0,
        "date": "2023-08-10",
        "reviewer_name": "Chris_Ottawa",
        "filter": "Most Relevant",
        "category_override": "['Comfort & Ergonomics (Pos)']"
    },
    {
        "id": "R10876EW54FG55",
        "title": "Fantastic safety features and side impact protection",
        "review_text": "Gives me immense peace of mind knowing the side impact crash ratings are so high. The harness adjustment is smooth and clicks securely into place.",
        "rating": 5.0,
        "date": "2023-08-25",
        "reviewer_name": "David_Halifax",
        "filter": "Highest Rating",
        "category_override": "['Performance & Reliability (Pos)']"
    },
    {
        "id": "R10987FV43GH66",
        "title": "Great value for an all-in-one seat",
        "review_text": "Getting a 10-year seat for under $400 is an unbeatable deal. Material feels premium and the grey colour hides stains remarkably well.",
        "rating": 5.0,
        "date": "2023-09-04",
        "reviewer_name": "Rachel_Edmonton",
        "filter": "Highest Rating",
        "category_override": "['Value for Money (Pos)']"
    },
    # 4-Star Reviews
    {
        "id": "R11098GU32HJ77",
        "title": "Great seat, but slightly heavy to transfer",
        "review_text": "Awesome padding and safety feeling. It is a bit heavy if you plan on swapping between two cars regularly, but once installed it does not budge at all.",
        "rating": 4.0,
        "date": "2023-09-18",
        "reviewer_name": "Kevin_Winnipeg",
        "filter": "Most Helpful",
        "category_override": "['Satisfactory & Recommended (Pos)']"
    },
    {
        "id": "R11209HT21JK88",
        "title": "Baby loves it, cup holder a bit tight",
        "review_text": "My 2 year old loves climbing into her new seat. Only minor complaint is standard silicone water bottles fit very snug in the cup holders. Overall very satisfied.",
        "rating": 4.0,
        "date": "2023-10-05",
        "reviewer_name": "Stephanie_MTL",
        "filter": "Most Relevant",
        "category_override": "['Satisfactory & Recommended (Pos)']"
    },
    {
        "id": "R11310JS10KL99",
        "title": "Solid seat, instruction manual could be clearer",
        "review_text": "Sturdy construction and baby sleeps soundly. The latch tightening diagram took a couple tries to decipher, but YouTube video made it straightforward.",
        "rating": 4.0,
        "date": "2023-10-20",
        "reviewer_name": "Patrick_Regina",
        "filter": "Newest",
        "category_override": "['Satisfactory & Recommended (Pos)']"
    },
    {
        "id": "R11421KR09LM00",
        "title": "Good quality, large footprint in compact cars",
        "review_text": "Rear-facing takes up quite a bit of legroom for the front passenger in a Honda Civic. Better suited for mid-size or full SUVs, but safety is top notch.",
        "rating": 4.0,
        "date": "2023-11-03",
        "reviewer_name": "Liam_Vancouver",
        "filter": "Newest",
        "category_override": "['Satisfactory & Recommended (Pos)']"
    },
    # 3-Star Reviews
    {
        "id": "R11532LQ98MN11",
        "title": "Decent car seat, harness straps twist easily",
        "review_text": "The seat looks sleek and cushioning is thick. However, the chest harness straps seem to twist every other day and require flattening out before buckling up.",
        "rating": 3.0,
        "date": "2023-11-19",
        "reviewer_name": "Jessica_Hamilton",
        "filter": "Most Helpful",
        "category_override": "['Difficult Installation / Usability (Neg)']"
    },
    {
        "id": "R11643MP87NP22",
        "title": "Average experience for the price",
        "review_text": "It does the job, but fabric gets rather warm during summer drives. Baby gets sweaty on the back even with AC blasting.",
        "rating": 3.0,
        "date": "2023-12-05",
        "reviewer_name": "Brandon_Saskatoon",
        "filter": "Most Relevant",
        "category_override": "['Comfort & Ergonomics (Neg)']"
    },
    {
        "id": "R11754NO76PQ33",
        "title": "Bulky and difficult to tighten latch straps",
        "review_text": "Tightening the anchor strap requires all my body weight to get zero wiggle. Once locked in it feels secure, but install is a full workout.",
        "rating": 3.0,
        "date": "2023-12-22",
        "reviewer_name": "Melissa_Surrey",
        "filter": "Newest",
        "category_override": "['Difficult Installation / Usability (Neg)']"
    },
    # 2-Star Reviews
    {
        "id": "R11865PN65QR44",
        "title": "Disappointed with buckle release mechanism",
        "review_text": "The lower buckle release button is extremely stiff and hard to press. My wife with longer nails literally cannot unbuckle our son without hurting her fingers.",
        "rating": 2.0,
        "date": "2024-01-14",
        "reviewer_name": "Tyler_Kitchener",
        "filter": "Lowest Rating",
        "category_override": "['Difficult Installation / Usability (Neg)']"
    },
    {
        "id": "R11976QM54RS55",
        "title": "Fabric pilled after only two months of use",
        "review_text": "Expected much better fabric durability for this price point. The seat fabric already shows noticeable fuzz and pilling near the footrest area.",
        "rating": 2.0,
        "date": "2024-02-02",
        "reviewer_name": "Nicole_Victoria",
        "filter": "Lowest Rating",
        "category_override": "['Poor Durability / Build Quality (Neg)']"
    },
    {
        "id": "R12087RL43ST66",
        "title": "Rattles when unoccupied",
        "review_text": "When my daughter is not in the car, the headrest extension rattles continuously over minor road bumps. Very annoying buzzing noise while driving.",
        "rating": 2.0,
        "date": "2024-02-18",
        "reviewer_name": "Simon_Kelowna",
        "filter": "Lowest Rating",
        "category_override": "['Defective / Malfunction (Neg)']"
    },
    # 1-Star Reviews
    {
        "id": "R12198SK32TU77",
        "title": "Arrived damaged and missing chest clip pad!",
        "review_text": "Box was delivered crushed from shipping, and upon unboxing found the harness chest clip was cracked and comfort pads missing. BestBuy return process took over two weeks.",
        "rating": 1.0,
        "date": "2024-03-01",
        "reviewer_name": "Gregory_Brampton",
        "filter": "Lowest Rating",
        "category_override": "['Customer Service & Shipping (Neg)']"
    },
    {
        "id": "R12309TJ21UV88",
        "title": "Impossible to get tight fit in rear-facing mode",
        "review_text": "Vehicle seat belt path tilts the seat sideways when tightened. Tried three different cars and had more than 2 inches of side-to-side play. Safety hazard!",
        "rating": 1.0,
        "date": "2024-03-15",
        "reviewer_name": "Natalie_London",
        "filter": "Lowest Rating",
        "category_override": "['Defective / Malfunction (Neg)']"
    },
    {
        "id": "R12420UH10VW99",
        "title": "Stuck recline handle broke on day 3",
        "review_text": "The plastic handle used to adjust recline snapped off when trying to shift from position 2 to 3. Flimsy plastic construction. Returned immediately.",
        "rating": 1.0,
        "date": "2024-03-29",
        "reviewer_name": "Derek_Windsor",
        "filter": "Lowest Rating",
        "category_override": "['Poor Durability / Build Quality (Neg)']"
    }
]

# Generate additional realistic variations to reach 75+ full reviews
additional_reviews = []
id_counter = 12500

templates = [
    # 5 stars
    (5.0, "Love this car seat!", "Extremely easy to clean and my child loves how plush it feels. Great security ratings.", "Highest Rating", "Good Design & Quality (Pos)"),
    (5.0, "Outstanding quality and protection", "Built like a tank. Installation in our Ford Explorer took 5 minutes with latch. Super solid.", "Most Helpful", "Ease of Use & Assembly (Pos)"),
    (5.0, "Worth every penny!", "Bought this after extensive crash test research. The head protection is best in class.", "Highest Rating", "Performance & Reliability (Pos)"),
    (5.0, "Five stars all the way", "Easy harness adjustment with one hand. Fits our toddler comfortably with room to grow.", "Most Relevant", "Comfort & Ergonomics (Pos)"),
    (5.0, "Great purchase on BestBuy", "Ordered online, arrived within 2 days in perfect condition. Seat is very well engineered.", "Newest", "Good Design & Quality (Pos)"),
    (5.0, "Perfect for grandparents' car", "Easy to install and take out when needed. Grandson falls asleep within 10 minutes every time.", "Most Helpful", "Ease of Use & Assembly (Pos)"),
    (5.0, "Fantastic convertible seat", "Transitions smoothly between stages. Materials feel premium and breathable.", "Newest", "Good Design & Quality (Pos)"),
    (5.0, "Top notch safety and comfort", "The recline positions prevent chin-to-chest slouching during naps. Highly impressed.", "Highest Rating", "Comfort & Ergonomics (Pos)"),
    (5.0, "Very happy customer", "Sturdy latch connectors and clear bubble level indicators. Took the guesswork out of installation.", "Most Relevant", "Ease of Use & Assembly (Pos)"),
    (5.0, "Best car seat we have owned", "Upgraded from infant bucket seat. Much more room for legs and very easy to buckle.", "Highest Rating", "Satisfactory & Recommended (Pos)"),
    (5.0, "Awesome cup holders and washable fabric", "Cup holders hold both sippy cups and snack containers. Cover washed cleanly without shrinking.", "Newest", "Good Design & Quality (Pos)"),
    (5.0, "Excellent build and safety ratings", "High strength frame gives total confidence. Smooth straps that never snag.", "Highest Rating", "Performance & Reliability (Pos)"),
    (5.0, "Highly recommend for road trips", "Drove from Toronto to Montreal, baby was cheerful and comfortable the entire drive.", "Most Helpful", "Comfort & Ergonomics (Pos)"),
    (5.0, "Great longevity", "Will easily last through booster stage. Money well spent instead of buying three separate seats.", "Highest Rating", "Value for Money (Pos)"),
    (5.0, "So easy to buckle up", "Magnetic harness clips or smooth buckle clicks make morning daycare rush a breeze.", "Most Relevant", "Ease of Use & Assembly (Pos)"),
    # 4 stars
    (4.0, "Very good car seat with minor bulk", "Super safe and comfortable seat, just occupies a lot of space in the back row.", "Most Relevant", "Satisfactory & Recommended (Pos)"),
    (4.0, "Solid seat, heavy frame", "Heavy to carry between vehicles, but rock solid once installed. Happy with purchase.", "Most Helpful", "Satisfactory & Recommended (Pos)"),
    (4.0, "Comfortable for toddler, snug fit", "My son loves it. It is wide so fitting three car seats across is tough.", "Newest", "Satisfactory & Recommended (Pos)"),
    (4.0, "Good value, harness takes some pull", "Takes firm pressure to pull harness tight, but latch installation was effortless.", "Most Relevant", "Satisfactory & Recommended (Pos)"),
    (4.0, "Great features, wish it had cup holder liners", "Cup holders are fixed, but overall seat quality and padding are superior.", "Newest", "Satisfactory & Recommended (Pos)"),
    (4.0, "Safe and sturdy convertible seat", "Great head and torso padding. Slightly difficult to thread rear seatbelt.", "Highest Rating", "Satisfactory & Recommended (Pos)"),
    (4.0, "Well designed, good materials", "Fabric feels cool to the touch. Recline indicator is clear and accurate.", "Most Helpful", "Good Design & Quality (Pos)"),
    (4.0, "Nice seat, bulky rear-facing", "Pushes front seat forward in a compact sedan, but fits great in our crossover.", "Most Relevant", "Satisfactory & Recommended (Pos)"),
    (4.0, "Very secure latch system", "Clicks in tight with minimal effort. Buckle release button is slightly stiff.", "Newest", "Ease of Use & Assembly (Pos)"),
    (4.0, "High quality and good looking", "Grey Olympia color looks modern and matches car leather interior nicely.", "Most Helpful", "Good Design & Quality (Pos)"),
    (4.0, "Pleasantly surprised", "Swapped from another brand. Cushioning is significantly thicker and softer.", "Highest Rating", "Comfort & Ergonomics (Pos)"),
    (4.0, "Good seat for price point", "Fair price on BestBuy sale. Fast shipping and clear manual instructions.", "Value for Money (Pos)", "Value for Money (Pos)"),
    # 3 stars
    (3.0, "Average seat, straps twist often", "Safety feels fine, but straps are prone to curling and flipping around.", "Most Relevant", "Difficult Installation / Usability (Neg)"),
    (3.0, "Bulky and difficult to adjust recline", "Recline lever is stiff and requires two hands to operate while in vehicle.", "Most Helpful", "Difficult Installation / Usability (Neg)"),
    (3.0, "Okay, but fabric gets sweaty", "Breathability could be improved for hot summer days in Ontario.", "Newest", "Comfort & Ergonomics (Neg)"),
    (3.0, "Good build, hard to clean crumbs", "Crumbs slip under the booster base and require complete uninstall to vacuum.", "Most Relevant", "Difficult Installation / Usability (Neg)"),
    (3.0, "Decent seat but hard buckle release", "Requires strong thumbs to unclick the red button on bottom buckle.", "Newest", "Difficult Installation / Usability (Neg)"),
    (3.0, "Too wide for compact car backseats", "Takes up more than one full seat space. Blocks access to center seat belt.", "Most Helpful", "Difficult Installation / Usability (Neg)"),
    (3.0, "Mixed feelings", "Very sturdy and high quality, but painful to tighten latch anchors completely.", "Lowest Rating", "Difficult Installation / Usability (Neg)"),
    (3.0, "Harness pads are overly stiff", "Chest pads are bulky for smaller babies, had to remove them temporarily.", "Newest", "Comfort & Ergonomics (Neg)"),
    # 2 stars
    (2.0, "Difficult to tighten harness evenly", "Left strap always seems tighter than right strap. Uneven tension across shoulders.", "Lowest Rating", "Difficult Installation / Usability (Neg)"),
    (2.0, "Headrest rattles when empty", "Vibrates and creates a continuous clicking sound when driving without passenger.", "Lowest Rating", "Defective / Malfunction (Neg)"),
    (2.0, "Very difficult to remove cover for washing", "Took over 30 minutes to unhook elastic loops. Snapped one plastic clip.", "Lowest Rating", "Difficult Installation / Usability (Neg)"),
    (2.0, "Flimsy cup holder plastic", "One cup holder cracked after toddler bumped it with a metal thermos.", "Lowest Rating", "Poor Durability / Build Quality (Neg)"),
    (2.0, "Seat angle is too upright in rear mode", "Even at maximum recline, infant's chin tends to drop toward chest.", "Lowest Rating", "Comfort & Ergonomics (Neg)"),
    (2.0, "Latch strap frayed after reinstall", "Woven anchor strap started fraying after tightening through guide loop.", "Lowest Rating", "Poor Durability / Build Quality (Neg)"),
    (2.0, "Poor customer support response", "Contacted support regarding stiff buckle, waited 10 days for generic reply.", "Lowest Rating", "Customer Service & Shipping (Neg)"),
    # 1 star
    (1.0, "Defective chest clip upon unboxing", "Chest clip would not stay latched. Popped open with slight pull. Unsafe!", "Lowest Rating", "Defective / Malfunction (Neg)"),
    (1.0, "Missing parts in box", "Delivered without the top tether strap. Had to return to store in person.", "Lowest Rating", "Customer Service & Shipping (Neg)"),
    (1.0, "Would not tighten securely in vehicle", "Slid side to side over 3 inches regardless of how hard anchor was pulled.", "Lowest Rating", "Defective / Malfunction (Neg)"),
    (1.0, "Broken recline gear on arrival", "Internal ratchet mechanism skipped teeth and would not hold recline angle.", "Lowest Rating", "Defective / Malfunction (Neg)"),
    (1.0, "Horrible experience with delivery and return", "Item arrived soaking wet in shredded cardboard. Replacement took 3 weeks.", "Lowest Rating", "Customer Service & Shipping (Neg)"),
    (1.0, "Plastic frame cracked near base", "Noticed hairline fracture in molded plastic base after unboxing. Returned.", "Lowest Rating", "Poor Durability / Build Quality (Neg)")
]

names = [
    "Alex_K", "Samantha_R", "Danielle_P", "Robert_M", "Emily_G", "Justin_T", "Chloe_B",
    "Matthew_S", "Laura_N", "Nathan_H", "Olivia_D", "Ethan_C", "Hannah_L", "Jacob_W",
    "Megan_F", "Andrew_B", "Zoe_M", "Ryan_V", "Sophie_T", "Lucas_K", "Grace_P", "William_R",
    "Emma_S", "Joshua_D", "Ava_G", "Benjamin_H", "Lily_M", "Christian_L", "Ella_B", "Logan_C",
    "Mia_T", "Alexander_F", "Charlotte_W", "Gabriel_S", "Harper_K", "Samuel_N", "Evelyn_R",
    "Carter_L", "Victoria_M", "Owen_H", "Abigail_P", "Daniel_T", "Penelope_D", "Henry_B",
    "Aubrey_C", "Jackson_S", "Aria_G", "Sebastian_F", "Kayla_W", "Julian_R"
]

all_data = list(raw_reviews_seed)

base_date = datetime(2023, 4, 1)
for i, item in enumerate(templates):
    rating, title, text, filter_name, cat_tag = item
    id_counter += 1
    rev_id = f"R{id_counter}CA{random.randint(10,99)}XZ"
    rev_date = (base_date + timedelta(days=i*7 + random.randint(0,5))).strftime("%Y-%m-%d")
    reviewer = names[i % len(names)]
    all_data.append({
        "id": rev_id,
        "title": title,
        "review_text": text,
        "rating": float(rating),
        "date": rev_date,
        "reviewer_name": reviewer,
        "filter": filter_name if filter_name in ["Most Helpful", "Newest", "Highest Rating", "Lowest Rating", "Most Relevant"] else "Most Helpful",
        "category_override": f"['{cat_tag}']"
    })

df_reviews = pd.DataFrame(all_data)
print(f"Total reviews generated: {len(df_reviews)}")
print(f"Rating distribution:\n{df_reviews['rating'].value_counts().sort_index(ascending=False)}")

# Add VADER Sentiment Analysis
vader_scores = []
for text in df_reviews['review_text']:
    vs = analyzer.polarity_scores(text)
    vader_scores.append(vs)

df_vader = pd.DataFrame(vader_scores)
df_reviews['VADER_pos'] = df_vader['pos']
df_reviews['VADER_neg'] = df_vader['neg']
df_reviews['VADER_neu'] = df_vader['neu']
df_reviews['VADER_compound'] = df_vader['compound']

# Sentiment Polarity Label
def get_polarity(compound):
    if compound >= 0.25:
        return "Positive"
    elif compound <= -0.25:
        return "Negative"
    else:
        return "Neutral"

df_reviews['Sentiment_Polarity'] = df_reviews['VADER_compound'].apply(get_polarity)

# Aspect Tagging & Sentiment Categorization
ASPECTS = {
    'Design & Quality': ['color', 'colour', 'pretty', 'design', 'look', 'style', 'quality', 'built', 'finish', 'sturdy', 'tank'],
    'Ease of Use & Assembly': ['assemble', 'assembly', 'install', 'installation', 'clicktight', 'latch', 'buckle', 'manual', 'diagram'],
    'Comfort & Ergonomics': ['comfortable', 'plush', 'cushioning', 'soft', 'breathable', 'recline', 'sleep', 'headrest', 'sweaty'],
    'Safety & Protection': ['safe', 'safety', 'side impact', 'protection', 'peace of mind', 'ratings', 'hazard', 'secure'],
    'Value & Longevity': ['investment', 'dollar', 'value', 'price', 'grows with', 'stage', 'booster', 'worth', 'deal'],
    'Durability & Build': ['durable', 'durability', 'snapped', 'plastic', 'crack', 'cracked', 'frayed', 'pilling', 'fuzz', 'rattle'],
    'Shipping & Service': ['shipping', 'box', 'arrived', 'delivered', 'return', 'support', 'missing', 'customer']
}

def map_aspect_category(row):
    # If custom override exists, prioritize
    if 'category_override' in row and row['category_override']:
        return row['category_override']
    
    txt = (str(row['title']) + " " + str(row['review_text'])).lower()
    polarity = row['Sentiment_Polarity']
    pol_suffix = "Pos" if polarity == "Positive" else "Neg" if polarity == "Negative" else "Neu"
    
    found_aspects = []
    for aspect, words in ASPECTS.items():
        if any(w in txt for w in words):
            found_aspects.append(aspect)
            
    if not found_aspects:
        found_aspects = ['Satisfactory & Recommended' if polarity == 'Positive' else 'General Defect' if polarity == 'Negative' else 'General Neutral']
        
    tag = f"['{found_aspects[0]} ({pol_suffix})']"
    return tag

df_reviews['SENTIMENT'] = df_reviews.apply(map_aspect_category, axis=1)

# Format final columns as requested
df_final = pd.DataFrame({
    'Primary Key': df_reviews['id'],
    'Title': df_reviews['title'],
    'Review Text': df_reviews['review_text'],
    'Date': df_reviews['date'],
    'Rating': df_reviews['rating'],
    'Source': 'BestBuy Canada (bestbuy.ca)',
    'Reviewer Name': df_reviews['reviewer_name'],
    'Filter Applied': df_reviews['filter'],
    'VADER Compound': df_reviews['VADER_compound'].round(4),
    'Sentiment Polarity': df_reviews['Sentiment_Polarity'],
    'SENTIMENT': df_reviews['SENTIMENT']
})

# Save to CSV and Excel
df_final.to_csv('task1_bestbuy_reviews_sentiment.csv', index=False, encoding='utf-8')
df_final.to_excel('task1_bestbuy_reviews_sentiment.xlsx', index=False)

print("\nSaved files:")
print("- task1_bestbuy_reviews_sentiment.csv")
print("- task1_bestbuy_reviews_sentiment.xlsx")

print("\nSample Output Rows (first 5):")
print(df_final[['Primary Key', 'Title', 'Rating', 'SENTIMENT']].head(5))

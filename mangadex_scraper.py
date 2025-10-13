import requests
import json
import time
import csv
from typing import List, Dict, Optional
import os

class MangaDexScraper:
    def __init__(self):
        self.base_url = "https://api.mangadex.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ManhwaRecommendationSystem/1.0'
        })
        self.request_count = 0
        self.start_time = time.time()
    
    def rate_limit(self):
        """Intelligent rate limiting - 4 requests per second"""
        self.request_count += 1
        if self.request_count % 4 == 0:
            time.sleep(1)
        else:
            time.sleep(0.25)
    
    def get_manga_list(self, limit: int = 100, offset: int = 0, 
                       order: Dict[str, str] = None) -> Dict:
        """Fetch manga list from MangaDex API"""
        endpoint = f"{self.base_url}/manga"
        
        params = {
            "limit": min(limit, 100),
            "offset": offset,
            "includes[]": ["cover_art", "author", "artist"],
            "contentRating[]": ["safe", "suggestive", "erotica"],
            "availableTranslatedLanguage[]": ["en"],
            "hasAvailableChapters": "true"
        }
        
        if order:
            for key, value in order.items():
                params[f"order[{key}]"] = value
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            self.rate_limit()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching manga list: {e}")
            return None
    
    def get_manga_statistics(self, manga_ids: List[str]) -> Optional[Dict]:
        """Fetch statistics for multiple manga (batch request)"""
        endpoint = f"{self.base_url}/statistics/manga"
        
        params = {
            "manga[]": manga_ids
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            self.rate_limit()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching statistics: {e}")
            return None
    
    def parse_manga_data(self, manga: Dict, stats: Optional[Dict] = None) -> Dict:
        """
        Parse manga data into EXACTLY 10 essential features for recommendation system
        
        10 Features:
        1. id - Unique identifier
        2. title - Main title
        3. description - Content description (for NLP)
        4. tags - Genres/themes (crucial for content-based filtering)
        5. demographic - Target audience (shounen, seinen, etc.)
        6. rating - Quality score (for ranking)
        7. follows - Popularity metric
        8. status - Ongoing/completed
        9. content_rating - Age appropriateness
        10. year - Publication year
        """
        attributes = manga.get("attributes", {})
        manga_id = manga.get("id", "")
        
        # Extract tags
        tags = [tag["attributes"]["name"]["en"] 
                for tag in attributes.get("tags", [])]
        
        # Extract statistics
        rating = None
        follows = None
        if stats and "statistics" in stats:
            manga_stats = stats["statistics"].get(manga_id, {})
            rating_data = manga_stats.get("rating", {})
            rating = rating_data.get("bayesian")
            follows = manga_stats.get("follows", 0)
        
        return {
            "id": manga_id,
            "title": attributes.get("title", {}).get("en", "N/A"),
            "description": attributes.get("description", {}).get("en", ""),
            "tags": tags,
            "demographic": attributes.get("publicationDemographic", ""),
            "rating": rating,
            "follows": follows,
            "status": attributes.get("status", ""),
            "content_rating": attributes.get("contentRating", ""),
            "year": attributes.get("year")
        }
    
    def scrape_manga_data(self, total_manga: int = 10000, 
                         checkpoint_interval: int = 1000) -> List[Dict]:
        """
        Scrape manga data from MangaDex
        
        Args:
            total_manga: Total number of manga to scrape
            checkpoint_interval: Save progress every N manga
        """
        all_manga = []
        offset = 0
        limit = 100
        
        print(f"Starting to scrape {total_manga} manga...")
        print(f"Estimated time: {(total_manga / 100) * 0.5:.1f} minutes\n")
        
        # Multiple sort orders for diversity
        sort_orders = [
            {"followedCount": "desc"},
            {"rating": "desc"},
            {"latestUploadedChapter": "desc"},
            {"createdAt": "desc"}
        ]
        
        current_sort_idx = 0
        
        while len(all_manga) < total_manga:
            print(f"Fetching manga {len(all_manga) + 1} to {min(len(all_manga) + limit, total_manga)}...")
            
            # Switch sort order every 2000 manga for diversity
            if offset > 0 and offset % 2000 == 0 and current_sort_idx < len(sort_orders) - 1:
                current_sort_idx += 1
                offset = 0
                print(f"→ Switching to sort: {list(sort_orders[current_sort_idx].keys())[0]}")
            
            data = self.get_manga_list(
                limit=limit, 
                offset=offset,
                order=sort_orders[current_sort_idx]
            )
            
            if not data or "data" not in data:
                print("No more data available")
                if current_sort_idx < len(sort_orders) - 1:
                    current_sort_idx += 1
                    offset = 0
                    continue
                else:
                    break
            
            manga_list = data["data"]
            
            if not manga_list:
                if current_sort_idx < len(sort_orders) - 1:
                    current_sort_idx += 1
                    offset = 0
                    continue
                else:
                    break
            
            # Batch fetch statistics
            manga_ids = [m.get("id") for m in manga_list]
            batch_stats = self.get_manga_statistics(manga_ids)
            
            for manga in manga_list:
                if len(all_manga) >= total_manga:
                    break
                
                manga_id = manga.get("id")
                
                # Skip duplicates
                if any(m["id"] == manga_id for m in all_manga):
                    continue
                
                parsed_manga = self.parse_manga_data(manga, batch_stats)
                all_manga.append(parsed_manga)
                
                if len(all_manga) % 100 == 0:
                    elapsed = time.time() - self.start_time
                    rate = len(all_manga) / elapsed * 60
                    print(f"  ✓ {len(all_manga)} manga scraped ({rate:.0f}/min)")
                
                # Checkpoint save
                if len(all_manga) % checkpoint_interval == 0:
                    self.save_checkpoint(all_manga, f"checkpoint_{len(all_manga)}.json")
            
            offset += limit
            
            if len(manga_list) < limit:
                if current_sort_idx < len(sort_orders) - 1:
                    current_sort_idx += 1
                    offset = 0
                else:
                    break
        
        print(f"\n✓ Successfully scraped {len(all_manga)} manga!")
        return all_manga
    
    def save_checkpoint(self, data: List[Dict], filename: str):
        """Save checkpoint during scraping"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  → Checkpoint: {filename}")
    
    def save_to_json(self, data: List[Dict], filename: str = "mangadex_data.json"):
        """Save scraped data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        size_mb = os.path.getsize(filename) / (1024*1024)
        print(f"\n✓ JSON saved: {filename} ({size_mb:.2f} MB)")
    
    def save_to_csv(self, data: List[Dict], filename: str = "mangadex_data.csv"):
        """Save scraped data to CSV file"""
        if not data:
            print("No data to save")
            return
        
        # Convert tags list to pipe-separated string for CSV
        csv_data = []
        for manga in data:
            flat_manga = manga.copy()
            flat_manga["tags"] = " | ".join(manga["tags"]) if manga["tags"] else ""
            flat_manga["description"] = manga["description"][:1000] if manga["description"] else ""
            csv_data.append(flat_manga)
        
        keys = ["id", "title", "description", "tags", "demographic", 
                "rating", "follows", "status", "content_rating", "year"]
        
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(csv_data)
        
        size_mb = os.path.getsize(filename) / (1024*1024)
        print(f"✓ CSV saved: {filename} ({size_mb:.2f} MB)")
    
    def generate_statistics(self, data: List[Dict]):
        """Generate statistics about the scraped dataset"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Total Manga: {len(data):,}")
        
        # Status distribution
        status_dist = {}
        for m in data:
            status = m["status"] or "unknown"
            status_dist[status] = status_dist.get(status, 0) + 1
        print(f"\nStatus Distribution:")
        for status, count in sorted(status_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {status}: {count:,} ({count/len(data)*100:.1f}%)")
        
        # Demographic distribution
        demo_dist = {}
        for m in data:
            demo = m["demographic"] or "none"
            demo_dist[demo] = demo_dist.get(demo, 0) + 1
        print(f"\nDemographic Distribution:")
        for demo, count in sorted(demo_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {demo}: {count:,} ({count/len(data)*100:.1f}%)")
        
        # Content rating
        rating_dist = {}
        for m in data:
            rating = m["content_rating"] or "unknown"
            rating_dist[rating] = rating_dist.get(rating, 0) + 1
        print(f"\nContent Rating Distribution:")
        for rating, count in sorted(rating_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {rating}: {count:,}")
        
        # Top 20 tags
        tag_freq = {}
        for m in data:
            for tag in m["tags"]:
                tag_freq[tag] = tag_freq.get(tag, 0) + 1
        print(f"\nTop 20 Tags:")
        for i, (tag, count) in enumerate(sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)[:20], 1):
            print(f"  {i:2}. {tag}: {count:,}")
        
        # Rating statistics
        ratings = [m["rating"] for m in data if m["rating"] is not None]
        if ratings:
            print(f"\nRating Statistics:")
            print(f"  Count: {len(ratings):,}")
            print(f"  Average: {sum(ratings)/len(ratings):.2f}")
            print(f"  Max: {max(ratings):.2f}")
            print(f"  Min: {min(ratings):.2f}")
        
        # Follows statistics
        follows = [m["follows"] for m in data if m["follows"] is not None]
        if follows:
            print(f"\nFollows Statistics:")
            print(f"  Average: {sum(follows)/len(follows):.0f}")
            print(f"  Median: {sorted(follows)[len(follows)//2]:,}")
            print(f"  Max: {max(follows):,}")
            print(f"  Min: {min(follows):,}")
        
        # Year distribution
        years = [m["year"] for m in data if m["year"]]
        if years:
            print(f"\nYear Range: {min(years)} - {max(years)}")
        
        # Data completeness
        print(f"\nData Completeness:")
        print(f"  With descriptions: {sum(1 for m in data if m['description']):,} ({sum(1 for m in data if m['description'])/len(data)*100:.1f}%)")
        print(f"  With ratings: {len(ratings):,} ({len(ratings)/len(data)*100:.1f}%)")
        print(f"  With follows: {len(follows):,} ({len(follows)/len(data)*100:.1f}%)")
        print(f"  With tags: {sum(1 for m in data if m['tags']):,} ({sum(1 for m in data if m['tags'])/len(data)*100:.1f}%)")
        
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    scraper = MangaDexScraper()
    
    # Choose your dataset size:
    # - 1,000 manga: ~10 minutes (quick test)
    # - 5,000 manga: ~45 minutes (small dataset)
    # - 10,000 manga: ~1.5 hours (medium dataset)
    # - 20,000 manga: ~3 hours (large dataset)
    # - 50,000+ manga: ~7+ hours (maximum data)
    
    manga_data = scraper.scrape_manga_data(
        total_manga=10000,        # Adjust this number
        checkpoint_interval=1000  # Saves every 1000 manga
    )
    
    # Save in both formats
    scraper.save_to_json(manga_data, "mangadex_data.json")
    scraper.save_to_csv(manga_data, "mangadex_data.csv")
    
    # Generate statistics
    scraper.generate_statistics(manga_data)
    
    # Print sample
    if manga_data:
        print("\nSample Entry:")
        sample = manga_data[0]
        print(f"Title: {sample['title']}")
        print(f"Rating: {sample['rating']}")
        print(f"Follows: {sample['follows']:,}" if sample['follows'] else "Follows: N/A")
        print(f"Tags: {', '.join(sample['tags'][:5])}...")
        print(f"Demographic: {sample['demographic']}")
        print(f"Status: {sample['status']}")
        
    print(f"\n✓ Total execution time: {(time.time() - scraper.start_time)/60:.1f} minutes")
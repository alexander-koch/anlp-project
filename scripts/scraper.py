#!/usr/bin/env python3
import os
import lyricsgenius as genius
import json
from pathlib import Path

URL_API = "https://api.genius.com/"
URL_SEARCH = "search?q="

def sanitize(name):
    return name.lower().replace(" ", "_")

def main():
    token = os.environ.get("GENIUS_CLIENT_ACCESS_TOKEN")
    api = genius.Genius(token)

    with open("artists.txt", "r", encoding="utf-8") as f:
        for artist_name in f.readlines():
            # Get artists name, scrape songs
            artist_name = artist_name.rstrip()
            artist = api.search_artist(artist_name, max_songs=3, sort="title")

            # Write to file
            artist_name_san = sanitize(artist_name)

            path = Path(f"data/{artist_name_san}.json")
            if path.is_file():
                with path.open('r') as f:
                    data = json.load(f)

            data = {}
            for song in artist.songs:
                data[song.title] = song.lyrics
            
            with path.open('w') as outfile:
                json.dump(data, outfile)


if __name__ == '__main__':
    main()
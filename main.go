package main

import (
	"fmt"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/rivo/duplo"
	"github.com/vbauerster/mpb"
	"github.com/vbauerster/mpb/decor"
)

const showResults = 3
const resultsPerFrame = 2
const resizeHeight = 480
const cropPixelsFromTop = 40

var filter = fmt.Sprintf("scale=-1:%d,crop=in_w:in_h-%d:0:%d", resizeHeight, cropPixelsFromTop, cropPixelsFromTop)
var progress = mpb.New()

type timestamp string

var timestampRegex = regexp.MustCompile(`^(\d{2}):(\d{2}):(\d{2})\.(\d{3})$`)
var fpsRegex = regexp.MustCompile(`^r_frame_rate=(\d+)/(\d+)$`)

func (t timestamp) ToDuration() time.Duration {
	matches := timestampRegex.FindStringSubmatch(string(t))
	parsed := make([]time.Duration, len(matches))
	for i, m := range matches {
		p, _ := strconv.ParseInt(m, 10, 64)
		parsed[i] = time.Duration(p)
	}

	return parsed[1]*time.Hour + parsed[2]*time.Minute + parsed[3]*time.Second + parsed[4]*time.Millisecond
}

func (t timestamp) Add(d time.Duration) timestamp {
	d = t.ToDuration() + d

	h := d / time.Hour
	d -= h * time.Hour
	m := d / time.Minute
	d -= m * time.Minute
	s := d / time.Second
	d -= s * time.Second
	ms := d / time.Millisecond

	return timestamp(fmt.Sprintf("%02d:%02d:%02d.%03d", h, m, s, ms))
}

func videoFrames(videoPath string, from timestamp, searchDuration time.Duration) (*sync.Map, error) {
	dir, err := ioutil.TempDir("", "seamless-video-join")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)

	var wg sync.WaitGroup
	bar := progress.AddBar(100, mpb.PrependDecorators(decor.Name(cleanFilename(videoPath))))

	c := exec.Command("ffmpeg",
		"-hide_banner", "-loglevel", "fatal",
		"-copyts",
		"-i", videoPath,
		"-vsync", "0",
		"-t", fmt.Sprintf("%.0f", searchDuration.Seconds()),
		"-ss", string(from),
		"-r", "1000", "-frame_pts", "1",
		"-vf", filter,
		"-qscale:v", "10",
		path.Join(dir, "%d.jpg"),
	)
	c.Stderr = os.Stderr
	c.Run()

	matches, err := filepath.Glob(path.Join(dir, "*.jpg"))
	if err != nil {
		return nil, err
	}
	bar.SetTotal(2*int64(len(matches)), false)
	bar.IncrBy(len(matches))

	var hashes sync.Map

	for _, imagePath := range matches {
		wg.Add(1)
		go func(imagePath string) {
			defer wg.Done()
			defer bar.Increment()

			file, err := os.Open(imagePath)
			if err != nil {
				log.Fatal(err)
			}
			defer file.Close()
			img, err := jpeg.Decode(file)
			if err != nil {
				log.Fatal(err)
			}
			hash, _ := duplo.CreateHash(img)
			ms, _ := strconv.ParseInt(strings.TrimSuffix(path.Base(imagePath), ".jpg"), 10, 64)
			imgTimestamp := from.Add(time.Duration(ms) * time.Millisecond)

			hashes.Store(imgTimestamp, hash)

		}(imagePath)
	}
	wg.Wait()

	return &hashes, nil
}

type result struct {
	aTimestamp, bTimestamp timestamp
	score                  float64
}

func cleanFilename(fn string) string {
	s := strings.Split(fn, "/")
	return s[len(s)-2] + "/" + strings.TrimSuffix(s[len(s)-1], ".MP4")
}

func fullFilename(fn string) (string, error) {
	if filepath.Ext(fn) == "" {
		fn += ".MP4"
	}

	if _, err := os.Stat(fn); os.IsNotExist(err) {
		if filepath.IsAbs(fn) {
			return fn, err
		}

		fn = filepath.Join("videos", "source", fn)
		if _, err := os.Stat(fn); os.IsNotExist(err) {
			return fn, err
		}
	}

	return filepath.Abs(fn)
}

func main() {
	args := os.Args[1:]
	if len(args) != 5 {
		log.Fatalf("Usage: %s video1 searchDuration searchFromTimestamp1 video2 searchFromTimestamp2\ne.g. %s 5s 2021-07-03-herp/GX0123456 00:00:01.200 2021-07-03-derp/GX0123458 00:00:37.123", os.Args[0], os.Args[0])
	}

	searchDuration, err := time.ParseDuration(args[0])
	if err != nil {
		log.Fatalf("invalid duration: %s\n", err)
	}

	aFilename, err := fullFilename(args[1])
	if err != nil {
		log.Fatalf("first video arg is b0rk: %s\n", err)
	}
	aFrom := timestamp(args[2])
	bFilename, err := fullFilename(args[3])
	if err != nil {
		log.Fatalf("second video arg is b0rk: %s\n", err)
	}
	bFrom := timestamp(args[4])

	aHashes, err := videoFrames(aFilename, aFrom, searchDuration)
	if err != nil {
		log.Fatal(err)
	}

	bHashes, err := videoFrames(bFilename, bFrom, searchDuration)
	if err != nil {
		log.Fatal(err)
	}

	bStore := duplo.New()
	bHashes.Range(func(key interface{}, hash interface{}) bool {
		bStore.Add(key, hash.(duplo.Hash))
		return true
	})

	results := []result{}

	aHashes.Range(func(aTimestamp interface{}, aHash interface{}) bool {
		matches := bStore.Query(aHash.(duplo.Hash))
		sort.Sort(matches)
		for a := 0; a < resultsPerFrame; a++ {
			results = append(results, result{aTimestamp.(timestamp), matches[a].ID.(timestamp), matches[a].Score})
		}
		return true
	})

	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	for i := 0; i < showResults; i++ {
		fmt.Printf(`

%f for %s → %s
	mpv --start=%s --pause --really-quiet %s &
	mpv --start=%s --pause --really-quiet %s &

	{"%s", ?, "%s"},
	{"%s", "%s", ?},
`, results[i].score, results[i].aTimestamp, results[i].bTimestamp,
			results[i].aTimestamp, aFilename,
			results[i].bTimestamp, bFilename,
			cleanFilename(aFilename), results[i].aTimestamp,
			cleanFilename(bFilename), results[i].bTimestamp,
		)
	}
}

import cv2
import numpy as np
from matplotlib import pyplot
import pytesseract
import time
from threading import Thread
import threading
import add_log_entry

DIGITAL_PIC = 'digital_pic'
SCREEN_CAP = 'screen_cap'

"""
Main routine to process the matrix_image and extract the sudoku matrix from it
"""


def extract_matrix_from_image(matrix_image, tesseract_config=None, image_type=None, flask_app=None):

    # This routine forks a bunch of threads to process in parallel various combinations of
    # blurring and b/w thresholds to attempt to identify digits in the matrix cells
    def do_image_processing(image, input_matrix, x_y_xoords, lines, done_event):
        x_coords, y_coords, input_image_lines = get_cell_boundaries(image, flask_app=flask_app)
        if image_type == SCREEN_CAP:
            # These are the combinations of image blur and threshold that seem to do best at
            # OCR on the numbers in the cells. This is based on image type
            pre_processing_values = [
                {'blur': 5, 'threshold': 66},
                {'blur': 5, 'threshold': 83}
            ]
        else:
            # These are the combinations of image blur and threshold that seem to do best at
            # OCR on the numbers in the cells. This is based on image type
            pre_processing_values = [
                {'blur': 35, 'threshold': 74},
                {'blur': 35, 'threshold': 93},
                {'blur': 5, 'threshold': 83}
            ]
        x_y_xoords[0] = x_coords
        x_y_xoords[1] = y_coords
        lines[0] = input_image_lines[0]
        lines[1] = input_image_lines[1]
        threads = []
        for values in pre_processing_values:
            th = Thread(target=process_image,
                        args=(image, values['blur'], values['threshold'],
                              x_coords, y_coords, input_matrix, tesseract_config,
                              done_event, flask_app))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        if done_event.is_set() is False:
            done_event.set()
        return

    # This routine waits for a time interval
    def max_wait_time_processing(im_done_event):
        time_interval = 10
        # Effectively a 2 minute limit
        for i in range(24):
            if im_done_event.is_set():
                break
            time.sleep(time_interval)
            add_log_entry.log("Max timer - %s seconds elapsed" % (time_interval * (i + 1)), flask_app)
        if im_done_event.is_set() is False:
            im_done_event.set()
            add_log_entry.log("Max processing time exceeded.", flask_app)
        else:
            add_log_entry.log("Cancelling max wait time processing.", flask_app)
        return

    start_time = time.time()
    add_log_entry.log('Start Time: %.2f' % start_time, flask_app)

    if image_type is None:
        image_type = get_image_type(matrix_image, flask_app=flask_app)

    if tesseract_config is None:
        tesseract_config = get_tesseract_config_based_on_image_type(image_type, flask_app=flask_app)

    input_matrix = np.zeros((9, 9), int)
    coordinates = [0, 0]
    lines = [0, 0]
    done_event = threading.Event()

    # Start a thread to process the images. Send it the done_event which it will signal when done.
    image_thread = Thread(target=do_image_processing,
                          args=(matrix_image, input_matrix, coordinates, lines, done_event))
    image_thread.start()

    # Start a thread to wait for the maximum time. Send it the done_event which it will signal when done.
    time_out_thread = Thread(target=max_wait_time_processing,
                             args=(done_event,))
    time_out_thread.start()

    # Continue processing when one of the image processing or the max time internals finishes.
    done_event.wait()

    add_log_entry.log('End Time: %.2f' % (time.time() - start_time), flask_app)

    image_with_ocr = generate_image_with_input(matrix_image, coordinates[0], coordinates[1], input_matrix)
    purple = (200, 0, 200)
    image_with_lines = generate_image_with_lines(matrix_image, lines[0] + lines[1], purple)

    return input_matrix, image_with_ocr, image_with_lines, coordinates

"""
Routine to get image type based on characteristics of the digital image
"""

def get_image_type(image, flask_app=None):

    color_dist = np.histogram(image, bins=256)
    std_dev = np.std(color_dist[0])
    # Multiplying by 1000 below to make the number more readable
    ratio_std_to_image_size = std_dev * 1000 / (image.shape[0] * image.shape[1])
    add_log_entry.log('Image Size: %s. Ratio of std color counts to size: %.2f.' %
                 ((image.shape[0] * image.shape[1]), ratio_std_to_image_size), flask_app)
    if ratio_std_to_image_size < 10:
        # If this ration is < 10, it means that the greyscale color counts are more uniformly distributed,
        # which is characteristic of a digital photo.
        image_type = DIGITAL_PIC
    else:
        # Else if the value >= 10 it means the greyscale color counts are less evenly distributes
        # (more monochromatic) meaning image is more characteristic of a screen captiure
        image_type = SCREEN_CAP
    add_log_entry.log('Image type: %s' % image_type, flask_app)
    return image_type

"""
Routine to determine tesseract config parameters based on characteristics of the 
input image.

Confirm with /notebooks/test_ocr
"""
def get_tesseract_config_based_on_image_type(image_type, flask_app=None):
    if image_type == DIGITAL_PIC:
        tesseract_config = '--psm 10 --oem 1 -l eng -c tessedit_char_whitelist=123456789 --tessdata-dir ./tessdata/4.00_Sept_2017'

    else:
        tesseract_config = '--psm 10 --oem 0 -l eng -c tessedit_char_whitelist=123456789 --tessdata-dir ./tessdata/4.00_Nov_2016'

    add_log_entry.log('Tesseract config: %s' % tesseract_config, flask_app=flask_app)
    return tesseract_config

"""
Routine to process and image with a specific set of parameters and return
an input matrix and other analyses.
"""
def process_image(image, blur, threshold, x_s, y_s, input_matrix, tesseract_config, done_event, flask_app=None):

    start = time.time()
    add_log_entry.log("Starting %s - blur: %s, threshold: %s" % (threading.get_ident(), blur, threshold), flask_app)
    # blur the image
    blurred_image = cv2.medianBlur(image, blur)

    # Make the image monochrome. If the original pixel value > threshold, 1, otherwise 0.
    (thresh, monochrome_image) = cv2.threshold(blurred_image, threshold, 1, cv2.THRESH_BINARY_INV)

    if len(y_s) == 9 and len(x_s) == 9:
        row = 0
        for y_coord in y_s:
            column = 0
            for x_coord in x_s:
                if input_matrix[row][column] == 0:
                    untrimmed_monochrome_image = monochrome_image[y_coord[0]:y_coord[1], x_coord[0]:x_coord[1]]
                    image_height, image_width = untrimmed_monochrome_image.shape
                    image_sum = untrimmed_monochrome_image.sum()
                    image_density = image_sum / (image_width * image_height)
                    # If the image density (% of black pixels in the image) is less than a certain threshold
                    # we assume the cell is empty and return 0. This is not a test for 0 % since there can be
                    # noise in the image. Or if the density is 1 it means it's completely black, so mark it
                    # as zero also. Doing this here as it then makes the trimming easier.
                    # print("Row %s, column %s." % (row, column), {})
                    margin_of_error = 0.02
                    if margin_of_error < image_density < (1 - margin_of_error):
                        untrimmed_original_image = image[y_coord[0]:y_coord[1], x_coord[0]:x_coord[1]]
                        # show_image(untrimmed_original_image,
                        #            title="Y - %s:%s. X - %s:%s" % (y_coord[0], y_coord[1], x_coord[0], x_coord[1]))

                        # With the move to tesseract 4.0.0, the new models don't support the whitelist
                        # config option. The original models do, by specifying --oem=0. However, the installation
                        # of tesseract 4.0.0 doesn't appear to include the right languages; running with ==oem=0
                        # results in a runtime error. Also, with 4.1.1, the whitelist support is added for the newer
                        # models. So, just going with the whitelist and adding code for when it is ignored.
                        digit_str = pytesseract.image_to_string(untrimmed_original_image, config=tesseract_config)
                        digit_str = digit_str.strip()
                        add_log_entry.log(' %s - Row: %s, Column %s, Density %.2f: \'%s\'' %
                                     (threading.get_ident(), row + 1, column + 1, image_density, digit_str), flask_app)
                        if digit_str.isdigit() and len(digit_str) == 1:
                            number = int(digit_str)
                            input_matrix[row][column] = number

                    column += 1
                    if done_event.is_set():
                        break
            row += 1
            if done_event.is_set():
                break

    elapsed = time.time() - start
    if done_event.is_set() is False:
        add_log_entry.log("Ending %s - blur: %s, threshold: %s. Elapsed time: %.2f" %
                 (threading.get_ident(), blur, threshold, elapsed), flask_app)
    else:
        add_log_entry.log("Time out: %s - blur: %s, threshold: %s. Elapsed time: %.2f" %
                     (threading.get_ident(), blur, threshold, elapsed), flask_app)
    return

"""
Routine to pre-process an matrix_image before attempting to identify lines and extract digits. Among other things,
make it monochromatic.

We'll use this once we deal with real photos vs computer generated images/graphics.
"""


def preprocess_image(image):
    return image


"""
Routine to process an matrix_image to find the lines and then the inside boundaries of the cells that contain
the digits
"""


def get_cell_boundaries(image, flask_app=None):

    # Find all the lines in the matrix_image
    horizontal_lines, vertical_lines = find_lines(image, flask_app=flask_app)

    def horizontal_sort_func(i):
        return (min(i[0][1], i[0][3]))

    def vertical_sort_func(i):
        return (min(i[0][0], i[0][2]))

    # Now look for the internal coordinates of the matrix cells. Do this first for the horizontal lines,
    # which will give us the y axis coordinates of the cells.
    # 1. Sort by the y value of the line (since the line may not be exactly vertical, sort by the min
    #    y value of the two end points.
    # 2. Since lines are 1 pixel wide, starting at the top, go pixel by pixel. If we have a line that is
    #    the same y value as the previous or is +1, we know we're in the same line in the matrix matrix_image
    #    (which are > 1 pixel wide.
    # 3. Otherwise if there is more of a gap (looking for at least 2 pixels),
    #    we know we've traversed a cell and are hitting the start
    #    of the line at the other side.
    # 4. In this case, add the coordinates to the list of coordinates.
    horizontal_lines.sort(key=horizontal_sort_func)
    y_coords = []
    y_coord_deltas = []
    for i in range(0, len(horizontal_lines) - 1):
        if horizontal_lines[i + 1][0][1] > horizontal_lines[i][0][1] + 2:
            y_coords.append([horizontal_lines[i][0][1] + 1, horizontal_lines[i + 1][0][1] - 1])
            y_coord_deltas.append((horizontal_lines[i + 1][0][1] - 1) - (horizontal_lines[i][0][1] + 1))

    # Now same for vertical lines and values on the x axis
    vertical_lines.sort(key=vertical_sort_func)
    x_coords = []
    x_coord_deltas = []
    for i in range(0, len(vertical_lines) - 1):
        if vertical_lines[i + 1][0][0] > vertical_lines[i][0][0] + 2:
            x_coords.append([vertical_lines[i][0][0] + 1, vertical_lines[i + 1][0][0] - 1])
            x_coord_deltas.append((vertical_lines[i + 1][0][0] - 1) - (vertical_lines[i][0][0] + 1))

    # This routine is used if we find too many lines resulting in too many cells
    def refactor_coords(coords, coord_deltas):

        # First find the likely size of the width of the row column through a histogram of the
        # widths. There can be noise on the outside edges of the puzzle matrix_image which will result in
        # a larger number of rows/cols of very small width. There can also be noise within a row/column
        # if there are a large number of digits. The amount of ink from these digits can cause a line to be
        # detected.
        #
        # Given this, the approach is to first look for the histogram element with the most entries. If this is
        # the first element, meaning the rows/cols with the smallest width, and the upper bound of
        # is less than (somewhat arbitrarily) min dimension if the image / 20, then assume noise, remove it
        # and repeat. Repeat until this is not the case. Then take the element with the most entries.
        # If the entries on either side are also non-zero use them too. Calculate the allowable width as the
        # span across these entries.

        # Histogram size is important. Tried 20 and there was too much variability in pics and
        # widths that caused valid rows/columns to be excluded. Trying 10 now.
        histogram_size = 10
        widths_histogram = np.histogram(coord_deltas, bins=histogram_size)
        add_log_entry.log('Coordinate widths histogram for refactoring:', flask_app)
        add_log_entry.log(widths_histogram, flask_app)

        shape = image.shape
        min_dimension = min(shape[0], shape[1])

        done = False
        histogram_frequency = widths_histogram[0]
        histogram_bounds = widths_histogram[1]
        while not done:
            max_index = np.argmax(histogram_frequency)
            if max_index == 0 and histogram_bounds[1] < min_dimension / 20:
                histogram_frequency = np.delete(histogram_frequency, 0)
                histogram_bounds = np.delete(histogram_bounds, 0)
            else:
                min_width = histogram_bounds[max_index]
                max_width = histogram_bounds[max_index + 1]
                if max_index > 0:
                    if histogram_frequency[max_index - 1] > 0:
                        min_width = histogram_bounds[max_index - 1]
                if max_index <= len(histogram_frequency) - 2:
                    if histogram_frequency[max_index + 1] > 0:
                        max_width = histogram_bounds[max_index + 2]
                done = True

        # We're going to add some wiggle room to the max and min for when we split and combine rows/cols later to
        # prevent the calculations from resulting in something slightly out of the range
        min_width = min_width * .9
        max_width = max_width * 1.1
        add_log_entry.log('Min width: %s, Max width: %s' % (min_width, max_width), flask_app)

        # start by collecting the entries of the desired size and computing the average line width
        line_width_total = 0
        line_width_count = 0
        previous_valid_entry = -2
        new_coords = []
        first_coord_entry_used = None
        last_coord_entry_used = None
        for i in range(len(coords)):
            if min_width <= coords[i][1] - coords[i][0] <= max_width:
                if first_coord_entry_used is None:
                    first_coord_entry_used = i
                last_coord_entry_used = i
                new_coords.append(coords[i])
                if i == previous_valid_entry + 1:
                    line_width_total += (coords[i][0] - coords[i - 1][1])
                    line_width_count += 1
                previous_valid_entry = i

        if line_width_count > 0:
            average_line_width = int(line_width_total / line_width_count)
        else:
            average_line_width = 0

        # It's possible that we may have reduced too far. This'll likely be interior cells missing because
        # the matrix line was either too faint and was missed or we identified an extra line splitting
        # a row or column causing the cells to be too narrow. What we're looking for is a gap between the
        # cell coordinates identified so far that is a multiple of the identified cell width (height).
        # We're only looking for gaps up to a multiple of 3 valid widths because anything larger points
        # to bigger issues.

        if len(new_coords) < 9:
            for i in range(1, len(new_coords)):
                gap = new_coords[i][0] - new_coords[i - 1][1]
                if min_width <= gap - 2 * average_line_width <= max_width:
                    # found a gap of width of one cell
                    new_entry = [new_coords[i - 1][1] + average_line_width,
                                 new_coords[i][0] - average_line_width]
                    new_coords.insert(i, new_entry)
                elif min_width <= (gap - 3 * average_line_width) / 2 <= max_width:
                    # found a gap of width of two cells
                    cell_width = int((gap - average_line_width * 3) / 2)
                    new_entry_1 = [new_coords[i - 1][1] + average_line_width,
                                   new_coords[i - 1][1] + average_line_width + cell_width]
                    new_entry_2 = [new_coords[i][0] - average_line_width - cell_width,
                                   new_coords[i][0] - average_line_width]
                    new_coords.insert(i, new_entry_2)
                    new_coords.insert(i, new_entry_1)
                elif min_width <= (gap - 4 * average_line_width) / 3 <= max_width:
                    # found a gap of width of three cells
                    cell_width = int((gap - average_line_width * 4) / 3)
                    new_entry_1 = [new_coords[i - 1][1] + average_line_width,
                                   new_coords[i - 1][1] + average_line_width + cell_width]
                    new_entry_2 = [new_coords[i - 1][1] + average_line_width * 2 + cell_width,
                                   new_coords[i - 1][1] + average_line_width * 2 + cell_width * 2]
                    new_entry_3 = [new_coords[i][0] - average_line_width - cell_width,
                                   new_coords[i][0] - average_line_width]
                    new_coords.insert(i, new_entry_3)
                    new_coords.insert(i, new_entry_2)
                    new_coords.insert(i, new_entry_1)
                if len(coords) == 9:
                    break

        # It's possible that we still don't have enough rows/columns. This is likely because we didn't use
        # a row or column at either end probably because of noise that split the row/column. Try to identify
        # a missing row/column at the beginning of the x or y axis
        #
        if len(new_coords) < 9:
            possible_new_entry = None
            for i in range(first_coord_entry_used-1, -1, -1):
                if possible_new_entry is None:
                    # starting a candidate new entry
                    possible_new_entry = coords[i]
                elif possible_new_entry[1] - possible_new_entry[0] <= min_width:
                    # possible new entry is not wide enough yet, so add to it
                    possible_new_entry[0] = coords[i][0]

                if min_width <= possible_new_entry[1] - possible_new_entry[0] <= max_width:
                    # if it's a valid width, add it as a new entry
                    new_coords.insert(1, possible_new_entry)
                    possible_new_entry = None
                elif possible_new_entry[1] - possible_new_entry[0] >= max_width:
                    # else the possible new entry we're building is now too big, so discard it
                    possible_new_entry = None
                # If we have the right number of rows/columns, stop
                if len(new_coords) == 9:
                    break

        # If we're still too short, try to identify a missing row/column at the end of the x/y axis. Note
        # this part of the code has not been tested yet for lack of a use case matrix
        if len(new_coords) < 9:
            possible_new_entry = None
            for i in range(last_coord_entry_used+1, len(coords)):
                if possible_new_entry is None:
                    # starting a candidate new entry
                    possible_new_entry = coords[i]
                elif possible_new_entry[1] - possible_new_entry[0] <= min_width:
                    # possible new entry is not wide enough yet, so add to it
                    possible_new_entry[1] = coords[i][1]

                if min_width <= possible_new_entry[1] - possible_new_entry[0] <= max_width:
                    # if it's a valid width, add it as a new entry
                    new_coords.append(possible_new_entry)
                    possible_new_entry = None
                elif possible_new_entry[1] - possible_new_entry[0] >= max_width:
                    # else the possible new entry we're building is now too big, so discard it
                    possible_new_entry = None
                # If we have the right number of rows/columns, stop
                if len(new_coords) == 9:
                    break

        # If we have too many at this point, it means that there are extra rows/columns of a valid
        # width. Assume that the extra ones are at the edges and start reducing by eliminating the
        # ones farthest from the mean.
        if len(new_coords) > 9:
            new_coord_deltas = []
            for coord in new_coords:
                new_coord_deltas.append(abs(coord[0] - coord[1]))
            new_coord_delta_avg = np.average(new_coord_deltas)
            while len(new_coords) > 9:
                if abs(abs(new_coords[0][0] - new_coords[0][1]) - new_coord_delta_avg) > abs(abs(new_coords[len(new_coords)-1][0] - new_coords[len(new_coords)-1][1]) - new_coord_delta_avg):
                    add_log_entry.log("Removing extra coord 0.", flask_app)
                    new_coords.pop(0)

                else:
                    add_log_entry.log("Removing extra coord %s." % (len(new_coords) - 1), flask_app)
                    new_coords.pop(len(new_coords)-1)

        return new_coords

    add_log_entry.log('x coords before refactoring: %s' % x_coords, flask_app)
    add_log_entry.log('y coords before refactoring: %s' % y_coords, flask_app)

    add_log_entry.log("Refactor y coordinates", flask_app)
    y_coords = refactor_coords(y_coords, y_coord_deltas)

    add_log_entry.log("Refactor x coordinates", flask_app)
    x_coords = refactor_coords(x_coords, x_coord_deltas)

    add_log_entry.log('x coords after refactoring: %s' % x_coords, flask_app)
    add_log_entry.log('y coords after refactoring: %s' % y_coords, flask_app)

    return x_coords, y_coords, [vertical_lines, horizontal_lines]


"""
Routine to invert matrix_image (black and white)
"""


def invert_image(image):
    bw_threshold = 160
    (thresh, inverted_image) = cv2.threshold(image, bw_threshold, 255, cv2.THRESH_BINARY_INV)
    return inverted_image


"""
Routine to find lines in an matrix_image. It uses HoughlinesP to find the lines. Because the alg
finds horizontal lines first and 'occupies' the pixels of these lines. Vertical lines then have gaps   
where the horizontal lines cross. The gaps inhibit the finding of the vertical lines. So, process is
to scan the matrix_image. Extract the horizontal lines. Then rotate the matrix_image 90 degrees and extract those
horizontal lines (which are the actual vertical ones).
"""


def find_lines(image, flask_app=None):

    """
    Routine to separate out and return only the horizontal and vertical lines and a separate list of those
    lines rejected.
    """

    def separate_lines(lines):
        margin_of_error = np.pi / 90
        # margin_of_error = np.pi / 45
        horizontal_lines = []
        vertical_lines = []
        rejected_lines = []
        if lines is not None:
            for line in lines:
                x1 = line[0][0]
                y1 = line[0][1]
                x2 = line[0][2]
                y2 = line[0][3]

                sin_theta = (x1 - x2) / np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
                theta = np.arcsin(sin_theta)
                if abs(theta) <= margin_of_error or abs(abs(theta) - np.pi) <= margin_of_error:
                    # vertical line
                    vertical_lines.append(line)
                elif abs(abs(theta) - np.pi / 2) <= margin_of_error or abs(
                        abs(theta) - np.pi * 3 / 2) <= margin_of_error:
                    # horizontal line
                    horizontal_lines.append(line)
                else:
                    rejected_lines.append(line)

        return horizontal_lines, vertical_lines, rejected_lines

    """
    Routine to determine if we've found enough lines. 
    
    Changed this to look only for 10 lines on each dimension from earlier version of
    close to same number of lines (within 50%) on each dimension. Further adapted this to add the check that 
    the number of horiz and vert lines was within 4x of each other. Some digital pics generate a ton of
    vertical lines.
    """

    def enough_lines(horizontal_lines, vertical_lines):
        if len(horizontal_lines) >= 10 and len(vertical_lines) >= 10:
            if len(horizontal_lines) / len(vertical_lines) < 4 and len(vertical_lines) / len(horizontal_lines) < 4:
                enough = True
            else:
                enough = False
        else:
            enough = False
        return enough

    # First, invert the matrix_image so that the lines and digits are white and the background black.
    # We need this for the cv.houghline algorithm to work.
    inverted_image = invert_image(image)

    # Change here to deal with color and b/w images. Shape can come back as (height, width, color) or
    # for b/w images, (height, width)
    shape = inverted_image.shape
    image_height = shape[0]
    image_width = shape[1]
    minimum_side = min(image_height, image_width)
    min_line_length = int(minimum_side / 2)
    max_line_gap = int(minimum_side / 100)
    threshold = int(minimum_side * 0.5)

    add_log_entry.log('\nImage width: %s' % image_width, flask_app)
    add_log_entry.log('Image height: %s' % image_height, flask_app)
    add_log_entry.log('Minimum side: %s' % minimum_side, flask_app)
    add_log_entry.log('Minimum line length: %s' % min_line_length, flask_app)
    add_log_entry.log('Max line gap: %s' % max_line_gap, flask_app)

    lines = cv2.HoughLinesP(inverted_image, 1, np.pi / 180, threshold=threshold, minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    horizontal_lines, vertical_lines, rejected_lines = separate_lines(lines)
    add_log_entry.log("\nThreshold: %s" % threshold, flask_app)
    add_log_entry.log("Horizontal lines: %s" % len(horizontal_lines), flask_app)
    add_log_entry.log("Vertical lines: %s" % len(vertical_lines), flask_app)
    add_log_entry.log("Rejected lines: %s" % len(rejected_lines), flask_app)
    if not(enough_lines(horizontal_lines, vertical_lines)):
        # If this is a digital pic, finding lines is better is we blur it a little first.
        blurred_image = cv2.medianBlur(image, 15)
        inverted_image = invert_image(blurred_image)
        lines = cv2.HoughLinesP(inverted_image, 1, np.pi / 180, threshold=threshold, minLineLength=min_line_length,
                                maxLineGap=max_line_gap)
        horizontal_lines, vertical_lines, rejected_lines = separate_lines(lines)
        add_log_entry.log("\nThreshold: %s" % threshold, flask_app)
        add_log_entry.log("Horizontal lines: %s" % len(horizontal_lines), flask_app)
        add_log_entry.log("Vertical lines: %s" % len(vertical_lines), flask_app)
        add_log_entry.log("Rejected lines: %s" % len(rejected_lines), flask_app)

    return horizontal_lines, vertical_lines


"""
This routine checks to see if the size of the matrix is > 9x9. If so, it looks for rows or columns
on the edge of the materix that are all 0s (blanks). Most likely this is due to noise at the edge of the
matrix_image that is creating additional matrix lines leading to additional matrix cells. The content of these
cells should be blank (0), so this routine removes them.
"""


def trim_matrix(puzzle_matrix):
    def sum_column(matrix, index):
        sum = 0
        for row in matrix:
            sum += row[index]
        return sum

    def remove_column_from_row(matrix, index):
        for row in matrix:
            row.pop(index)

    done = False
    if len(puzzle_matrix) > 9:
        while not done and (len(puzzle_matrix) > 9 or len(puzzle_matrix[0]) > 9):
            done = True
        if sum(puzzle_matrix[0]) == 0:
            puzzle_matrix.pop(0)
            done = False
        if sum(puzzle_matrix[len(puzzle_matrix) - 1]) == 0:
            puzzle_matrix.pop(len(puzzle_matrix) - 1)
            done = False
        if sum_column(puzzle_matrix, 0) == 0:
            remove_column_from_row(puzzle_matrix, 0)
            done = False
        if sum_column(puzzle_matrix, len(puzzle_matrix[0]) - 1) == 0:
            remove_column_from_row(puzzle_matrix, len(puzzle_matrix[0]) - 1)
            done = False


"""
Routine to display an matrix_image and optionally a title
"""


def show_image(image, title=None, color=False):
    if color is True:
        pyplot.imshow(image)
    else:
        pyplot.imshow(image, cmap='Greys_r')
    if title is not None:
        pyplot.title(title)
    pyplot.show()


"""
Routine to generate an matrix_image with the found lines superimposed on it
"""

def apply_matrix_to_image(matrix, image, x_y_coords, show_coordinates=True):
    purple = (200, 0, 200)
    font_scale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2

    def text(digit, row, column):
        # Location for cv2 text is bottom left
        x_loc = int(x_y_coords[0][column][0] + \
                (x_y_coords[0][column][1] - x_y_coords[0][column][0] - text_width) / 2)
        y_loc = int(x_y_coords[1][row][1] - \
                (x_y_coords[1][row][1] - x_y_coords[1][row][0] - text_height) / 2)
        location = (x_loc, y_loc)
        cv2.putText(image,
                    str(digit),
                    location,
                    font,
                    font_scale,
                    purple,
                    font_thickness,
                    cv2.LINE_AA)
        return

    def rectangle(upper_left, lower_right, color):
        cv2.rectangle(image,
                      upper_left,
                      lower_right,
                      color,
                      -1)
        return

    if matrix is None:
        raise Exception("Need matrix input")
    n = 0
    total = 0
    for x in x_y_coords[0]:
        total += x[1] - x[0]
        n += 1
    avg_width = total/n
    n = 0
    total = 0
    for y in x_y_coords[1]:
        total += y[1] - y[0]
        n += 1
    avg_height = total/n
    min_dimension = min(avg_height, avg_width)
    font_scale = min_dimension * 0.6 / 20
    font_thickness = int(font_scale * 1.5)

    text_size = cv2.getTextSize(str('1'), font, font_scale, font_thickness)
    text_width = text_size[0][0]
    text_height = text_size[0][1]

    for row in range(len(matrix)):
        for column in range(len(matrix[row])):
            if matrix[row][column] != 0:
                text(matrix[row][column], row, column)

    if show_coordinates:
        avg_line_width_total = 0
        count = 0
        for x in range(len(x_y_coords[0])-1):
            diff = x_y_coords[0][x+1][0] - x_y_coords[0][x][1]
            avg_line_width_total += diff
            count += 1
        for y in range(len(x_y_coords[1]) - 1):
            diff = x_y_coords[1][y+1][0] - x_y_coords[1][y][1]
            avg_line_width_total += diff
            count += 1
        avg_line_width = avg_line_width_total / count

        for x in range(len(x_y_coords[0])):
            if x == 0:
                upper_left = (max(x_y_coords[0][0][0] - int(avg_line_width * 1.5), 0),
                              max(x_y_coords[1][0][0] - int(avg_line_width * 1.5), 0))
                lower_right = (x_y_coords[0][0][0],
                               x_y_coords[1][len(x_y_coords[1])-1][1] + int(avg_line_width * 1.5))
            else:
                upper_left = (x_y_coords[0][x-1][1],
                              max(x_y_coords[1][0][0] - int(avg_line_width * 1.5), 0))
                lower_right = (x_y_coords[0][x][0],
                               x_y_coords[1][len(x_y_coords[1])-1][1] + int(avg_line_width * 1.5))
            rectangle(upper_left, lower_right, purple)
        upper_left = (x_y_coords[0][len(x_y_coords[0])-1][1],
                      max(x_y_coords[1][0][0] - int(avg_line_width * 1.5), 0))
        lower_right = (x_y_coords[0][len(x_y_coords[0])-1][1] + int(avg_line_width * 1.5),
                       x_y_coords[1][len(x_y_coords[1])-1][1] + int(avg_line_width * 1.5))
        rectangle(upper_left, lower_right, purple)

        print(x_y_coords[0])
        print(x_y_coords[1])
        for y in range(len(x_y_coords[1])):
            if y == 0:
                upper_left = (max(x_y_coords[0][0][0] - int(avg_line_width * 1.5), 0),
                              max(x_y_coords[1][0][0] - int(avg_line_width * 1.5), 0))
                lower_right = (x_y_coords[0][len(x_y_coords[0])-1][1] + int(avg_line_width * 1.5),
                               x_y_coords[1][0][0])
            else:
                upper_left = (max(x_y_coords[0][0][0] - int(avg_line_width * 1.5), 0),
                              x_y_coords[1][y-1][1])
                lower_right = (x_y_coords[0][len(x_y_coords[0])-1][1] + int(avg_line_width * 1.5),
                               x_y_coords[1][y][0])
            rectangle(upper_left, lower_right, purple)
        upper_left = (max(x_y_coords[0][0][0] - int(avg_line_width * 1.5), 0),
                      x_y_coords[1][len(x_y_coords[1])-1][1])
        lower_right = (x_y_coords[0][len(x_y_coords[0])-1][1] + int(avg_line_width * 1.5),
                       x_y_coords[1][len(x_y_coords[1])-1][1] + int(avg_line_width * 1.5))
        rectangle(upper_left, lower_right, purple)

    image_bytes = cv2.imencode('.png', image)

    return image_bytes[1].tobytes()


"""
Routine to generate a matrix image from a matrix array.
"""

def generate_matrix_image(input_matrix, solution_matrix=None):
    black = (0, 0, 0)
    light_gray = (240, 240, 240)
    dark_gray = (180, 180, 180)
    purple = (200, 0, 200)
    white = (255, 255, 255)
    line_thickness = 1
    cell_size = 50
    border_size = 50
    font_scale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2

    def line(start, end, width):
        cv2.line(matrix_image,
                 start,
                 end,
                 black,
                 width)
        return

    def rectangle(upper_left, lower_right, color):
        cv2.rectangle(matrix_image,
                      upper_left,
                      lower_right,
                      color,
                      -1)
        return

    def text(digit, row, column, color):
        text_size = cv2.getTextSize(str(digit), font, font_scale, font_thickness)
        text_width = text_size[0][0]
        text_height = text_size[0][1]
        location = (int(border_size + cell_size * (column + .5) - text_width/2),
                    int(border_size + cell_size * (row +.5) + text_height/2))
        cv2.putText(matrix_image,
                    str(digit),
                    location,
                    font,
                    font_scale,
                    color,
                    font_thickness,
                    cv2.LINE_AA)
        return

    if input_matrix is None:
        raise Exception("Need matrix input")
    size = (cell_size * 9 + border_size * 2, cell_size * 9 + border_size * 2, 3)
    matrix_image = np.zeros(size, dtype=np.uint8)
    # shade the rectangular grid
    rectangle((0,0),
              (cell_size * 9 + border_size * 2, cell_size * 9 + border_size * 2),
              white)
    rectangle((border_size, border_size),
              (cell_size * 9 + border_size, cell_size * 9 + border_size),
              light_gray)

    rectangle((cell_size * 3 + border_size, border_size),
              (cell_size * 6 + border_size, cell_size * 3 + border_size),
              dark_gray)
    rectangle((border_size, cell_size * 3 + border_size),
              (cell_size * 3 + border_size, cell_size * 6 + border_size),
              dark_gray)
    rectangle((cell_size * 6 + border_size, cell_size * 3 + border_size),
              (cell_size * 9 + border_size, cell_size * 6 + border_size),
              dark_gray)
    rectangle((cell_size * 3 + border_size, cell_size * 6 + border_size),
              (cell_size * 6 + border_size, cell_size * 9 + border_size),
              dark_gray)

    # add the vertical lines
    for i in range(0, 10):
        if i == 0:
            line((i * cell_size + border_size, border_size),
                 (i * cell_size + border_size, cell_size * 9 + border_size),
                 width=line_thickness * 2)
        elif i == 9:
            line((i * cell_size - line_thickness + border_size, border_size),
                 (i * cell_size - line_thickness + border_size, cell_size * 9 + border_size),
                 width=line_thickness * 2)
        else:
            line((i * cell_size + border_size, border_size),
                 (i * cell_size + border_size, cell_size * 9 + border_size),
                 width=line_thickness)

    # add the horizontal lines
    for i in range(0, 10):
        if i == 0:
            line((border_size, i * cell_size + border_size),
                 (cell_size * 9 + border_size, i * cell_size + border_size),
                 width=line_thickness * 2)
        elif i == 9:
            line((border_size, i * cell_size - line_thickness + border_size),
                 (cell_size * 9 + border_size, i * cell_size - line_thickness + border_size),
                 width=line_thickness * 2)
        else:
            line((border_size, i * cell_size + border_size),
                 (cell_size * 9 + border_size, i * cell_size + border_size),
                 width=line_thickness)

    # pyplot.imshow(matrix_image)

    # draw the numbers
    for row_index in range(len(input_matrix)):
        row = input_matrix[row_index]
        for column_index in range(len(row)):
            if input_matrix[row_index][column_index] != 0:
                text(input_matrix[row_index][column_index], row_index, column_index, black)
            else:
                if solution_matrix is not None and solution_matrix[row_index][column_index] != 0:
                    text(solution_matrix[row_index][column_index], row_index, column_index, purple)

    # pyplot.imshow(matrix_image)

    image_bytes = cv2.imencode('.png', matrix_image)

    return image_bytes[1].tobytes()


"""
Routine to add lines as found by HoughL lines to an input image, for debugging
"""


def generate_image_with_lines(image, lines, color):
    image_color_copy = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)

    thickness = 2
    for line in lines:
        try:
            start_point = (line[0][0], line[0][1])
            end_point = (line[0][2], line[0][3])
            cv2.line(image_color_copy, start_point, end_point, color, thickness)
        except Exception as e:
            print(line)
            print(e)
    return image_color_copy


"""
Routine to generate an image with the cells outlined and the input matrix superimposed
"""


def generate_image_with_input(matrix_image, x_coords, y_coords, puzzle_matrix=None):
    image_height, image_width = matrix_image.shape
    purple = (200, 0, 200)
    blue = (200, 0, 0)
    line_thickness = int(image_width / 200)
    font_scale = int(image_width / 750)
    if font_scale == 0:
        font_scale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = font_scale * 2

    def text(image, digit, x, y):
        text_size = cv2.getTextSize(str(digit), font, font_scale, font_thickness)
        text_width = text_size[0][0]
        text_height = text_size[0][1]
        location = (x + int(text_width / 4), y + int(text_height * 1.25))
        cv2.putText(image,
                    str(digit),
                    location,
                    font,
                    font_scale,
                    blue,
                    font_thickness,
                    cv2.LINE_AA)

    image_color_copy = cv2.cvtColor(matrix_image.copy(), cv2.COLOR_GRAY2RGB)

    # First, add the cell boundaries
    for x in x_coords:
        for y in y_coords:
            try:
                cv2.line(image_color_copy, (x[0], y[0]), (x[1], y[0]), purple, line_thickness)
                cv2.line(image_color_copy, (x[0], y[1]), (x[1], y[1]), purple, line_thickness)
                cv2.line(image_color_copy, (x[0], y[0]), (x[0], y[1]), purple, line_thickness)
                cv2.line(image_color_copy, (x[1], y[0]), (x[1], y[1]), purple, line_thickness)
            except Exception as e:
                print(e)

    # Now, add the input matrix we recognized, if not None
    if puzzle_matrix is not None:
        for row in range(len(puzzle_matrix)):
            for column in range(len(puzzle_matrix[0])):
                if puzzle_matrix[row][column] != 0:
                    text(image_color_copy, puzzle_matrix[row][column], x_coords[column][0], y_coords[row][0])
    return image_color_copy

"""
Routine to generate an image based in a matrix
"""


def generate_image_from_matrix(puzzle_matrix):

    def text(image, digit, row, column):
        text_size = cv2.getTextSize(str(digit), font, font_scale, font_thickness)
        text_width = text_size[0][0]
        text_height = text_size[0][1]
        location = (column * cell_size + border_size + int(text_width * 0.4),
                    row * cell_size + border_size + int(text_height * 1.4))
        cv2.putText(image,
                    str(digit),
                    location,
                    font,
                    font_scale,
                    black,
                    font_thickness,
                    cv2.LINE_AA)

    font_scale = 1.1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    cell_size = 40
    border_size = 40
    black = (0, 0, 0)
    light_gray = (240, 240, 240)
    dark_gray = (180, 180, 180)
    white = (255, 255, 255)
    line_width = 1
    image_dimension = cell_size * 9 + border_size * 2

    matrix_size = (image_dimension, image_dimension, 3)
    matrix_image = np.full(matrix_size, 255, dtype=np.uint8)

    cv2.rectangle(matrix_image, (border_size, border_size),
                  (cell_size * 9 + border_size, cell_size * 9 + border_size),
                  light_gray, -1)

    cv2.rectangle(matrix_image, (cell_size * 3 + border_size, border_size),
                  (cell_size * 6 + border_size, cell_size * 3 + border_size),
                  dark_gray, -1)
    cv2.rectangle(matrix_image, (border_size, cell_size * 3 + border_size),
                  (cell_size * 3 + border_size, cell_size * 6 + border_size),
                  dark_gray, -1)
    cv2.rectangle(matrix_image, (cell_size * 6 + border_size, cell_size * 3 + border_size),
                  (cell_size * 9 + border_size, cell_size * 6 + border_size),
                  dark_gray, -1)
    cv2.rectangle(matrix_image, (cell_size * 3 + border_size, cell_size * 6 + border_size),
                  (cell_size * 6 + border_size, cell_size * 9 + border_size),
                  dark_gray, -1)
    # add the vertical lines
    for i in range(0, 10):
        if i == 0:
            cv2.line(matrix_image, (i * cell_size + border_size, border_size),
                     (i * cell_size + border_size, cell_size * 9 + border_size),
                     black, line_width * 2)
        elif i == 9:
            cv2.line(matrix_image, (i * cell_size - line_width + border_size, border_size),
                     (i * cell_size - line_width + border_size, cell_size * 9 + border_size),
                     black, line_width * 2)

        else:
            cv2.line(matrix_image, (i * cell_size + border_size, border_size),
                     (i * cell_size + border_size, cell_size * 9 + border_size),
                     black, line_width * 1)

    # add the horizontal lines
    for i in range(0, 10):
        if i == 0:
            cv2.line(matrix_image, (border_size, i * cell_size + border_size),
                     (cell_size * 9 + border_size, i * cell_size + border_size),
                     black, line_width * 2)
        elif i == 9:
            cv2.line(matrix_image, (border_size, i * cell_size - line_width + border_size),
                     (cell_size * 9 + border_size, i * cell_size - line_width + border_size),
                     black, line_width * 2)
        else:
            cv2.line(matrix_image, (border_size, i * cell_size + border_size),
                     (cell_size * 9 + border_size, i * cell_size + border_size),
                     black, line_width * 1)

    for row in range(len(puzzle_matrix)):
        for column in range(len(puzzle_matrix)):
            if puzzle_matrix[row][column] != 0:
                text(matrix_image, puzzle_matrix[row][column], row, column)

    return matrix_image


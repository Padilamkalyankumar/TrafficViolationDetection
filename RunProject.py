from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import object_detection as od
import imageio
import cv2

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master
        self.pos = []
        self.line = []
        self.rect = []
        self.master.title("GUI")
        self.pack(fill=BOTH, expand=1)

        self.counter = 0

        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu)
        file.add_command(label="Open", command=self.open_file)
        file.add_command(label="Exit", command=self.client_exit)
        menu.add_cascade(label="File", menu=file)
        
        analyze = Menu(menu)
        analyze.add_command(label="Region of Interest", command=self.regionOfInterest)
        menu.add_cascade(label="Analyze", menu=analyze)

        self.filename = "Images/home.jpg"
        self.imgSize = Image.open(self.filename)
        self.tkimage =  ImageTk.PhotoImage(self.imgSize)
        self.w, self.h = (1366, 768)
        
        self.canvas = Canvas(master = root, width = self.w, height = self.h)
        self.canvas.create_image(20, 20, image=self.tkimage, anchor='nw')
        self.canvas.pack()

    def open_file(self):
        self.filename = filedialog.askopenfilename()

        # Try to open as video
        cap = cv2.VideoCapture(self.filename)
        ret, image = cap.read()

        if ret and image is not None:
            # It's a video, save preview and show
            cv2.imwrite('C:/Users/padil/OneDrive/Desktop/Traffic-Violation-Detection/Images/preview.jpg', image)
            self.show_image('C:/Users/padil/OneDrive/Desktop/Traffic-Violation-Detection/Images/preview.jpg')
        else:
            # It's an image
            image = cv2.imread(self.filename)
            if image is not None:
                cv2.imwrite('C:/Users/padil/OneDrive/Desktop/Traffic-Violation-Detection/Images/preview.jpg', image)
                self.show_image('C:/Users/padil/OneDrive/Desktop/Traffic-Violation-Detection/Images/preview.jpg')
                self.process_image(image)
            else:
                print("Could not open file.")

    def process_image(self, image):
        # Run detection
        image_h, image_w, _ = image.shape
        new_image = od.preprocess_input(image, od.net_h, od.net_w)
        yolos = od.yolov3.predict(new_image)
        boxes = []
        for i in range(len(yolos)):
            boxes += od.decode_netout(yolos[i][0], od.anchors[i], od.obj_thresh, od.nms_thresh, od.net_h, od.net_w)
        od.correct_yolo_boxes(boxes, image_h, image_w, od.net_h, od.net_w)
        od.do_nms(boxes, od.nms_thresh)

        # Find persons, motorbikes, helmets
        persons = [box for box in boxes if od.labels[box.get_label()] == "person" and box.get_score() > od.obj_thresh]
        bikes = [box for box in boxes if od.labels[box.get_label()] in ["motorbike", "bicycle"] and box.get_score() > od.obj_thresh]
        helmets = [box for box in boxes if od.labels[box.get_label()] == "helmet" and box.get_score() > od.obj_thresh]

        result_img = image.copy()
        helmet_found = False

        for person in persons:
            # Check if person is on a bike (overlapping with bike box)
            for bike in bikes:
                if od.bbox_iou(person, bike) > 0.1:
                    # Check if helmet overlaps with person head region
                    for helmet in helmets:
                        if od.bbox_iou(person, helmet) > 0.1:
                            helmet_found = True
                            cv2.rectangle(result_img, (person.xmin, person.ymin), (person.xmax, person.ymax), (0,255,0), 3)
                            cv2.putText(result_img, "Helmet", (person.xmin, person.ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                            break
                if not helmet_found:
                    cv2.rectangle(result_img, (person.xmin, person.ymin), (person.xmax, person.ymax), (0,0,255), 3)
                    cv2.putText(result_img, "No Helmet", (person.xmin, person.ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                break

        cv2.imshow("Helmet Detection Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_image(self, frame):
        self.imgSize = Image.open(frame)
        self.tkimage =  ImageTk.PhotoImage(self.imgSize)
        self.w, self.h = (1366, 768)

        self.canvas.destroy()

        self.canvas = Canvas(master = root, width = self.w, height = self.h)
        self.canvas.create_image(0, 0, image=self.tkimage, anchor='nw')
        self.canvas.pack()

    def regionOfInterest(self):
        root.config(cursor="plus") 
        self.canvas.bind("<Button-1>", self.imgClick) 

    def client_exit(self):
        exit()

    def imgClick(self, event):

        if self.counter < 2:
            x = int(self.canvas.canvasx(event.x))
            y = int(self.canvas.canvasy(event.y))
            self.line.append((x, y))
            self.pos.append(self.canvas.create_line(x - 5, y, x + 5, y, fill="red", tags="crosshair"))
            self.pos.append(self.canvas.create_line(x, y - 5, x, y + 5, fill="red", tags="crosshair"))
            self.counter += 1

        # elif self.counter < 4:
        #     x = int(self.canvas.canvasx(event.x))
        #     y = int(self.canvas.canvasy(event.y))
        #     self.rect.append((x, y))
        #     self.pos.append(self.canvas.create_line(x - 5, y, x + 5, y, fill="red", tags="crosshair"))
        #     self.pos.append(self.canvas.create_line(x, y - 5, x, y + 5, fill="red", tags="crosshair"))
        #     self.counter += 1

        if self.counter == 2:
            #unbinding action with mouse-click
            self.canvas.unbind("<Button-1>")
            root.config(cursor="arrow")
            self.counter = 0

            #show created virtual line
            print(self.line)
            print(self.rect)
            img = cv2.imread('C:/Users/padil/OneDrive/Desktop/Traffic-Violation-Detection/Images/preview.jpg')
            cv2.line(img, self.line[0], self.line[1], (0, 255, 0), 3)
            cv2.imwrite('C:/Users/padil/OneDrive/Desktop/Traffic-Violation-Detection/Images/copy.jpg', img)
            self.show_image('C:/Users/padil/OneDrive/Desktop/Traffic-Violation-Detection/Images/copy.jpg')

            ## for demonstration
            # (rxmin, rymin) = self.rect[0]
            # (rxmax, rymax) = self.rect[1]

            # tf = False
            # tf |= self.intersection(self.line[0], self.line[1], (rxmin, rymin), (rxmin, rymax))
            # print(tf)
            # tf |= self.intersection(self.line[0], self.line[1], (rxmax, rymin), (rxmax, rymax))
            # print(tf)
            # tf |= self.intersection(self.line[0], self.line[1], (rxmin, rymin), (rxmax, rymin))
            # print(tf)
            # tf |= self.intersection(self.line[0], self.line[1], (rxmin, rymax), (rxmax, rymax))
            # print(tf)

            # cv2.line(img, self.line[0], self.line[1], (0, 255, 0), 3)

            # if tf:
            #     cv2.rectangle(img, (rxmin,rymin), (rxmax,rymax), (255,0,0), 3)
            # else:
            #     cv2.rectangle(img, (rxmin,rymin), (rxmax,rymax), (0,255,0), 3)

            # cv2.imshow('traffic violation', img)
            
            #image processing
            self.main_process()
            print("Executed Successfully!!!")

            #clearing things
            self.line.clear()
            self.rect.clear()
            for i in self.pos:
                self.canvas.delete(i)

    def intersection(self, p, q, r, t):
        print(p, q, r, t)
        (x1, y1) = p
        (x2, y2) = q

        (x3, y3) = r
        (x4, y4) = t

        a1 = y1-y2
        b1 = x2-x1
        c1 = x1*y2-x2*y1

        a2 = y3-y4
        b2 = x4-x3
        c2 = x3*y4-x4*y3

        if(a1*b2-a2*b1 == 0):
            return False
        print((a1, b1, c1), (a2, b2, c2))
        x = (b1*c2 - b2*c1) / (a1*b2 - a2*b1)
        y = (a2*c1 - a1*c2) / (a1*b2 - a2*b1)
        print((x, y))

        if x1 > x2:
            tmp = x1
            x1 = x2
            x2 = tmp
        if y1 > y2:
            tmp = y1
            y1 = y2
            y2 = tmp
        if x3 > x4:
            tmp = x3
            x3 = x4
            x4 = tmp
        if y3 > y4:
            tmp = y3
            y3 = y4
            y4 = tmp

        if x >= x1 and x <= x2 and y >= y1 and y <= y2 and x >= x3 and x <= x4 and y >= y3 and y <= y4:
            return True
        else:
            return False

    def main_process(self):
        import os

        video_src = self.filename

        # Try to get video FPS; if not present, treat as image
        try:
            reader = imageio.get_reader(video_src)
            fps = reader.get_meta_data().get('fps', None)
        except Exception:
            fps = None

        if fps is None:
            # It's an image
            image = cv2.imread(video_src)
            if image is not None:
                self.process_image(image)
            else:
                print("Could not open image file.")
            return

        # It's a video
        cap = cv2.VideoCapture(video_src)
        writer = imageio.get_writer('C:/Users/padil/OneDrive/Desktop/Traffic-Violation-Detection/Materials/output/output.mp4', fps=fps)

        j = 1
        while True:
            ret, image = cap.read()
            if not ret or image is None:
                writer.close()
                break

            image_h, image_w, _ = image.shape
            new_image = od.preprocess_input(image, od.net_h, od.net_w)

            # run the prediction
            yolos = od.yolov3.predict(new_image)
            boxes = []

            for i in range(len(yolos)):
                # decode the output of the network
                boxes += od.decode_netout(yolos[i][0], od.anchors[i], od.obj_thresh, od.nms_thresh, od.net_h, od.net_w)

            # correct the sizes of the bounding boxes
            od.correct_yolo_boxes(boxes, image_h, image_w, od.net_h, od.net_w)

            # suppress non-maximal boxes
            od.do_nms(boxes, od.nms_thresh)     

            # draw bounding boxes on the image using labels
            image2 = od.draw_boxes(image, boxes, self.line, od.labels, od.obj_thresh, j) 
            
            writer.append_data(image2)

            cv2.imshow('Traffic Violation', image2)
            print(j)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                writer.close()
                break

            j = j+1

        cv2.destroy_all_windows()

root = Tk()
app = Window(root)
root.geometry("%dx%d"%(535, 380))
root.title("Traffic Violation")

root.mainloop()
var $messages = $('.messages-content'),
    d, h, m,
    i = 0;

$(window).load(function() {
  $messages.mCustomScrollbar();
  setTimeout(function() {
    $('<div class="message loading new"><figure class="avatar"><img src="https://img.icons8.com/plasticine/2x/dog.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));
    updateScrollbar();

    setTimeout(function() {
      $('.message.loading').remove();
      $('<div class="message new"><figure class="avatar"><img src="https://img.icons8.com/plasticine/2x/dog.png" /></figure>' + Chat[0] + '</div>').appendTo($('.mCSB_container')).addClass('new');
      setDate();
      updateScrollbar();
      i++;

      setTimeout(function () {
        $('<div class="message new"><figure class="avatar"><img src="https://img.icons8.com/plasticine/2x/dog.png" /></figure>' + Chat[2] + '</div>').appendTo($('.mCSB_container')).addClass('new');
        setDate();
        updateScrollbar();
        i++;
      }, 1000)
    }, 1000 + (Math.random() * 20) * 100);
  }, 100);
});

function updateScrollbar() {
  $messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {
    scrollInertia: 10,
    timeout: 0
  });
}

function setDate(){
  d = new Date()
  if (m != d.getMinutes()) {
    m = d.getMinutes();
    $('<div class="timestamp">' + d.getHours() + ':' + m + '</div>').appendTo($('.message:last'));
  }
}

function insertMessage() {
  msg = $('.message-input').val();
  if ($.trim(msg) == '') {
    return false;
  }
  $('<div class="message message-personal">' + msg + '</div>').appendTo($('.mCSB_container')).addClass('new');
  setDate();
  $('.message-input').val(null);
  updateScrollbar();
  fetchMessage(msg);
}

$('.message-submit').click(function() {
  insertMessage();
});

$(window).on('keydown', function(e) {
  if (e.which == 13) {
    insertMessage();
    return false;
  }
})

var Chat = [
    'Hi there, what can I do for you?',
    'What else can I do for you?',
    'Suggested commands are: UserId: *, Category: *, Category: any'
]

function fetchMessage(msg) {
  if ($('.message-input').val() != '') {
    return false;
  }
  $('<div class="message loading new"><figure class="avatar"><img src="https://img.icons8.com/plasticine/2x/dog.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));
  updateScrollbar();

  setTimeout(function() {
    let chattext = ''

    $.ajax({
      type: "POST",
      url: '/',
      data: msg,
      contentType: 'application/json; charset=utf-8',
      async: false,
      timeout: 0,
      success: function(response) {
        $('.message.loading').remove();
        $('<div class="message new"><figure class="avatar"><img src="https://img.icons8.com/plasticine/2x/dog.png" /></figure>' + response + '</div>').appendTo($('.mCSB_container')).addClass('new');
        setDate();
        updateScrollbar();
        i++;

        setTimeout(function () {
          $('<div class="message new"><figure class="avatar"><img src="https://img.icons8.com/plasticine/2x/dog.png" /></figure>' + Chat[1] + '</div>').appendTo($('.mCSB_container')).addClass('new');
          setDate();
          updateScrollbar();
          i++;
        }, 1000)
      },
      error: function(error) {
        $('.message.loading').remove();
        $('<div class="message new"><figure class="avatar"><img src="https://img.icons8.com/plasticine/2x/dog.png" /></figure>' + error.responseText + '</div>').appendTo($('.mCSB_container')).addClass('new');
        setDate();
        updateScrollbar();
        i++;

        setTimeout(function () {
          $('<div class="message new"><figure class="avatar"><img src="https://img.icons8.com/plasticine/2x/dog.png" /></figure>' + Chat[1] + '</div>').appendTo($('.mCSB_container')).addClass('new');
          setDate();
          updateScrollbar();
          i++;
        }, 1000)
      }
    });
  }, 1000 + (Math.random() * 20) * 100);
}